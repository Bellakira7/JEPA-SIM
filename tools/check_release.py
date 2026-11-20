"""Read-only syntax, source-only inventory and private-reference scan."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def inspect(root):
    allowed = {'.py', '.md', '.txt', '.yaml', '.yml', '.toml', '.json'}
    ignored = {'__pycache__', '.pytest_cache', '.git', '.venv', 'venv', 'data', 'artifacts',
               'checkpoints', 'outputs', 'runs', 'third_party'}
    special = {'.gitignore', '.gitattributes'}
    # Reject workspace storage prefixes, internal services and credential literals.
    patterns = {
        'private_storage': re.compile(r'/cpfs\d*/|[A-Z]:[\\/](?:Users|workspace)[\\/]', re.I),
        'internal_service': re.compile(r'https?://[^\s/]+\.(?:internal|intra|corp)(?:/|\b)', re.I),
        'credential_literal': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'),
    }
    findings, inventory, python_count = [], [], 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in ignored for part in path.relative_to(root).parts[:-1]):
            continue
        data = path.read_bytes()
        inventory.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if path.suffix not in allowed and path.name not in special:
            findings.append(dict(path=relative, check='unexpected_file_type'))
            continue
        text = data.decode('utf-8')
        if path.suffix == '.py':
            python_count += 1
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as error:
                findings.append(dict(path=relative, line=error.lineno, check='python_syntax'))
        elif path.suffix == '.json':
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                findings.append(dict(path=relative, line=error.lineno, check='json_syntax'))
        for check, pattern in patterns.items():
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    # Never echo matched credential or service text.
                    findings.append(dict(path=relative, line=number, check=check))
    return dict(status='pass' if not findings else 'fail', files=len(inventory), python_files=python_count,
                bytes=sum(item['bytes'] for item in inventory), findings=findings, inventory=inventory)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--report', type=Path, help='Write outside the source directory')
    args = ap.parse_args()
    root = args.root.resolve()
    if args.report and args.report.resolve().is_relative_to(root):
        raise ValueError('Keep review reports outside the source candidate')
    result = inspect(root)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({key: value for key, value in result.items() if key != 'inventory'}, indent=2))
    raise SystemExit(0 if result['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()
