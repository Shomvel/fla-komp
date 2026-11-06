# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check archival hashes and numerical-source equivalence after package relocation."""

import ast
import hashlib
import json
from pathlib import Path


class Normalize(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        # autopep8 rewrites assignment lambdas as equivalent local functions.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Lambda):
            function = ast.FunctionDef(
                name=node.targets[0].id,
                args=node.value.args,
                body=[ast.Return(value=node.value.body)],
                decorator_list=[],
                returns=None,
            )
            return self.visit_FunctionDef(function)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return node


def verify(root):
    manifest = json.loads((root / "research/komp/source-manifest.json").read_text())
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise AssertionError(f"Archive link escapes repository: {path}")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    count = 0
    for entry in manifest["active"]:
        if entry["target"] == "fla/ops/komp/__init__.py":
            continue
        source = root / "research/komp/history" / entry["source"]
        target = root / entry["target"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["source_sha256"]
        original = Normalize().visit(ast.parse(source.read_text()))
        imported = Normalize().visit(ast.parse(target.read_text()))
        assert ast.dump(original) == ast.dump(imported), entry["target"]
        count += 1
    return dict(
        archive_files=len(manifest["files"]),
        unique_contents=len({e["sha256"] for e in manifest["files"]}),
        equivalent_operator_files=count,
        upstream_commit=manifest["upstream_commit"],
    )


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
