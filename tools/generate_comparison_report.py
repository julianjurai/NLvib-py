#!/usr/bin/env python3
"""
Generate MATLAB vs Python comparison report from comparison notebooks.

Usage:
    python tools/generate_comparison_report.py

Outputs:
    - docs/comparison_report.md
    - docs/images/comparison/*.png (extracted from notebooks)
"""

import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "examples" / "comparison"
IMG_DIR = ROOT / "docs" / "images" / "comparison"
MD_OUT = ROOT / "docs" / "comparison_report.md"


def extract_notebook_data(nb_path: Path) -> Dict[str, Any]:
    """Extract metrics, images, and validation results from a comparison notebook."""

    nb = nbformat.read(nb_path, as_version=4)

    data = {
        'notebook': nb_path.stem,
        'name': '',
        'description': '',
        'matlab_source': '',
        'peak_error_pct': None,
        'metrics': {},
        'harmonics': {},
        'validation_status': 'UNKNOWN',
        'runtime_python': None,
        'runtime_matlab': None,
        'speedup': None,
        'images': [],
    }

    # Parse cells
    for cell_idx, cell in enumerate(nb.cells):
        # Extract description from first markdown cell
        if cell.cell_type == 'markdown' and not data['name']:
            source = ''.join(cell.source)
            # Look for "Example XX — Title"
            match = re.search(r'Example\s+(\d+)\s*[—-]\s*(.+)', source, re.IGNORECASE)
            if match:
                data['name'] = f"Example {match.group(1)}: {match.group(2).strip()}"
                # Extract MATLAB source reference
                ref_match = re.search(r'\*\*Reference\*\*:\s*`(.+?)`', source)
                if ref_match:
                    data['matlab_source'] = ref_match.group(1)

        # Extract outputs
        for out_idx, out in enumerate(cell.get('outputs', [])):
            # Text outputs (metrics, validation results)
            if out.output_type in ('stream', 'execute_result'):
                text = out.get('text', '') if out.output_type == 'stream' else out.get('data', {}).get('text/plain', '')
                if isinstance(text, list):
                    text = ''.join(text)

                # Look for peak error
                match = re.search(r'Peak\s+(?:amplitude\s+)?error[:\s=]+([0-9.]+)%', text, re.IGNORECASE)
                if match:
                    data['peak_error_pct'] = float(match.group(1))

                # Look for PASS/FAIL
                if re.search(r'\bPASS\b', text):
                    data['validation_status'] = 'PASS'
                elif re.search(r'\bFAIL\b', text):
                    data['validation_status'] = 'FAIL'

                # Extract metrics table
                metrics_match = re.findall(
                    r'^\s*(Peak\s+\w+.*?)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.]+)%',
                    text, re.MULTILINE
                )
                for m in metrics_match:
                    metric_name = m[0].strip()
                    data['metrics'][metric_name] = {
                        'matlab': float(m[1]),
                        'python': float(m[2]),
                        'diff': float(m[3]),
                        'error_pct': float(m[4]),
                    }

                # Extract runtime
                py_time = re.search(r'Python\s+(?:HB\s+)?wall\s+time\s*:\s*([0-9.]+)\s*s', text, re.IGNORECASE)
                if py_time:
                    data['runtime_python'] = float(py_time.group(1))

                matlab_time = re.search(r'(?:Octave|MATLAB)\s+wall\s+time\s*:\s*([0-9.]+)\s*s', text, re.IGNORECASE)
                if matlab_time:
                    data['runtime_matlab'] = float(matlab_time.group(1))

                speedup = re.search(r'Speedup\s+\([^)]*\)\s*:\s*([0-9.]+)x', text, re.IGNORECASE)
                if speedup:
                    data['speedup'] = float(speedup.group(1))

            # Image outputs
            png_data = out.get('data', {}).get('image/png')
            if png_data:
                # Decode and save image
                IMG_DIR.mkdir(parents=True, exist_ok=True)
                img_bytes = base64.b64decode(png_data.replace('\n', ''))
                img_filename = f"{nb_path.stem}_img{len(data['images'])}.png"
                img_path = IMG_DIR / img_filename
                img_path.write_bytes(img_bytes)
                data['images'].append(img_filename)

    return data


def generate_summary_table(all_data: List[Dict[str, Any]]) -> str:
    """Generate summary table of all examples."""

    lines = [
        "## Summary",
        "",
        f"Total examples validated: **{len(all_data)}**",
        "",
        "| Example | Peak Error | Status | Runtime (Python) | Runtime (MATLAB) | Speedup |",
        "|---------|------------|--------|------------------|------------------|---------|",
    ]

    for data in all_data:
        name = data['name'] or data['notebook']
        error = f"{data['peak_error_pct']:.3f}%" if data['peak_error_pct'] is not None else "—"
        status = "✅ PASS" if data['validation_status'] == 'PASS' else "❌ FAIL" if data['validation_status'] == 'FAIL' else "?"
        rt_py = f"{data['runtime_python']:.1f}s" if data['runtime_python'] else "—"
        rt_ml = f"{data['runtime_matlab']:.1f}s" if data['runtime_matlab'] else "—"
        speedup = f"{data['speedup']:.1f}x" if data['speedup'] else "—"

        lines.append(f"| {name} | {error} | {status} | {rt_py} | {rt_ml} | {speedup} |")

    lines.append("")
    return "\n".join(lines)


def generate_example_section(data: Dict[str, Any]) -> str:
    """Generate detailed section for one example."""

    lines = [
        f"## {data['name'] or data['notebook']}",
        "",
    ]

    # MATLAB source reference
    if data['matlab_source']:
        lines.append(f"**MATLAB Reference**: `{data['matlab_source']}`")
        lines.append("")

    # Visual comparison images
    if data['images']:
        lines.append("### Visual Comparison")
        lines.append("")
        for img in data['images']:
            lines.append(f"![{data['name']}](images/comparison/{img})")
            lines.append("")

    # Metrics table
    if data['metrics']:
        lines.append("### Metrics")
        lines.append("")
        lines.append("| Metric | MATLAB | Python | |Diff| | Rel.Err% |")
        lines.append("|--------|-------:|-------:|-------:|---------:|")
        for metric_name, vals in data['metrics'].items():
            lines.append(
                f"| {metric_name} | {vals['matlab']:.6g} | {vals['python']:.6g} | "
                f"{vals['diff']:.6g} | {vals['error_pct']:.3f}% |"
            )
        lines.append("")

    # Validation status
    lines.append("### Validation Status")
    lines.append("")
    if data['validation_status'] == 'PASS':
        if data['peak_error_pct'] is not None:
            lines.append(f"✅ **PASS** — Peak error: {data['peak_error_pct']:.3f}%")
        else:
            lines.append("✅ **PASS**")
    elif data['validation_status'] == 'FAIL':
        if data['peak_error_pct'] is not None:
            lines.append(f"❌ **FAIL** — Peak error: {data['peak_error_pct']:.3f}%")
        else:
            lines.append("❌ **FAIL**")
    else:
        lines.append("⚠️  **UNKNOWN** — Validation status not determined")
    lines.append("")

    # Runtime comparison
    if data['runtime_python'] or data['runtime_matlab']:
        lines.append("### Runtime")
        lines.append("")
        if data['runtime_python']:
            lines.append(f"- **Python HB**: {data['runtime_python']:.2f}s")
        if data['runtime_matlab']:
            lines.append(f"- **MATLAB/Octave**: {data['runtime_matlab']:.2f}s")
        if data['speedup']:
            faster = "Python" if data['speedup'] > 1 else "MATLAB"
            lines.append(f"- **Speedup**: {data['speedup']:.1f}x ({faster} faster)")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def generate_report(notebooks: List[Path], output_md: Path):
    """Generate complete markdown report from all comparison notebooks."""

    print(f"Extracting data from {len(notebooks)} notebooks...")
    all_data = []
    for nb_path in sorted(notebooks):
        print(f"  - {nb_path.name}")
        data = extract_notebook_data(nb_path)
        all_data.append(data)

    print(f"\nGenerating report...")

    # Build markdown content
    lines = [
        "# MATLAB vs Python Validation Report",
        "",
        "*Auto-generated from comparison notebooks in `examples/comparison/`*",
        "",
        f"*Last updated: {Path.cwd().name}*",
        "",
        "This report validates the Python implementation of the NLvib harmonic balance solver",
        "against the original MATLAB/Octave reference code. All 8 examples run both implementations",
        "with identical parameters and compare frequency response curves, peak amplitudes, and",
        "numerical accuracy.",
        "",
    ]

    # Summary table
    lines.append(generate_summary_table(all_data))

    # Per-example sections
    for data in all_data:
        lines.append(generate_example_section(data))

    # Appendix
    lines.extend([
        "## Validation Methodology",
        "",
        "Each comparison notebook:",
        "",
        "1. **Runs MATLAB/Octave reference** via subprocess, saves data to `.mat` file",
        "2. **Runs Python harmonic balance** continuation with identical parameters",
        "3. **Overlays both curves** on a single figure for visual comparison",
        "4. **Computes numerical metrics**: peak amplitude error, peak frequency error",
        "5. **Asserts validation**: Python must match MATLAB within specified tolerance (<1% or <5%)",
        "",
        "**Metrics:**",
        "- Peak amplitude error: relative difference at maximum response amplitude",
        "- Peak frequency error: relative difference at peak response frequency",
        "- Harmonic content: Fourier coefficients at fundamental and higher harmonics",
        "",
        "**Tolerances:**",
        "- Examples 01-04, 07: < 1% peak error",
        "- Examples 05, 06, 08: < 5% peak error (due to Galerkin truncation or hysteretic elements)",
        "",
        "---",
        "",
        "## Reference",
        "",
        "Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*. Springer.",
        "[https://doi.org/10.1007/978-3-030-14023-6](https://doi.org/10.1007/978-3-030-14023-6)",
        "",
    ])

    # Write output
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text('\n'.join(lines))

    print(f"\n✅ Report generated: {output_md}")
    print(f"✅ Images extracted: {len(list(IMG_DIR.glob('*.png')))} files in {IMG_DIR}")


if __name__ == '__main__':
    notebooks = list(NB_DIR.glob('[0-9]*.ipynb'))  # Only numbered notebooks
    if not notebooks:
        print(f"No comparison notebooks found in {NB_DIR}")
        exit(1)

    generate_report(notebooks, MD_OUT)
