"""Run the original FADING/HRFAE CLI in one model-resident child process.

Only heavyweight construction/checkpoint loading is memoized. The original
argument parsing, preprocessing, inversion, editing and image saving execute
again for every request. Each request gets a fresh globals dictionary and (for
FADING) its original fresh scheduler. This file imports no project packages so
the upstream repositories keep their isolated import namespaces.
"""

import ast
import gc
import json
import os
from pathlib import Path
import sys
import traceback


RESULT_PREFIX = "__AGING_RESIDENT_RESULT__"


class CacheModelConstruction(ast.NodeTransformer):
    def __init__(self, name):
        self.target = "ldm_stable" if name == "FADING" else "trainer"
        self.count = 0

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == self.target for target in node.targets):
            self.count += 1
            factory = ast.Lambda(
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=node.value,
            )
            node.value = ast.Call(func=ast.Name(id="_resident_model", ctx=ast.Load()), args=[factory], keywords=[])
        return node

    def visit_Expr(self, node):
        call = node.value
        if (self.target == "trainer" and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute) and call.func.attr in {"load_checkpoint", "to"}
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "trainer"):
            node.value = ast.Call(
                func=ast.Name(id="_resident_checkpoint" if call.func.attr == "load_checkpoint" else "_resident_place", ctx=ast.Load()),
                args=[call.func.value, *call.args], keywords=call.keywords,
            )
        return node


class ResidentScript:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.model_key = None
        self.checkpoint_key = None
        self.placement_key = None
        self.compiled = {}

    def run(self, cmd, cwd=None):
        script = Path(cmd[1]).resolve()
        if script not in self.compiled:
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            transform = CacheModelConstruction(self.name)
            tree = transform.visit(tree)
            if transform.count != 1:
                raise RuntimeError(f"Unsupported {self.name} runner: expected one {transform.target} construction in {script}")
            self.compiled[script] = compile(ast.fix_missing_locations(tree), str(script), "exec")
        old_cwd, old_argv, old_path = os.getcwd(), sys.argv, list(sys.path)
        namespace = {"__name__": "__main__", "__file__": str(script), "__package__": None}

        def get_model(factory):
            key = (str(script), str(namespace.get("specialized_path"))) if self.name == "FADING" else (
                str(script), json.dumps(namespace.get("config"), sort_keys=True, default=str)
            )
            if self.model is None or self.model_key != key:
                self.model = None
                gc.collect()
                self.model = factory()
                self.model_key = key
                self.checkpoint_key = None
                self.placement_key = None
            elif self.name == "FADING":
                # The CLI constructs this scheduler before loading the pipeline.
                # Never carry timesteps/history from the previous image.
                self.model.scheduler = namespace["scheduler"]
            return self.model

        def load_checkpoint(model, *args, **kwargs):
            key = (id(model), repr(args), repr(sorted(kwargs.items())))
            if key != self.checkpoint_key:
                model.load_checkpoint(*args, **kwargs)
                self.checkpoint_key = key

        def place(model, *args, **kwargs):
            key = (id(model), repr(args), repr(sorted(kwargs.items())))
            if key != self.placement_key:
                model.to(*args, **kwargs)
                self.placement_key = key

        namespace.update(_resident_model=get_model, _resident_checkpoint=load_checkpoint, _resident_place=place)
        try:
            if cwd:
                os.chdir(cwd)
            sys.path.insert(0, str(script.parent))
            sys.argv = list(map(str, cmd[1:]))
            exec(self.compiled[script], namespace)
        finally:
            # Do not retain gradients created by FADING's null-text optimization.
            for module in [self.model, *(getattr(self.model, key, None) for key in ("unet", "vae", "text_encoder"))]:
                if module is not None and hasattr(module, "zero_grad"):
                    module.zero_grad(set_to_none=True)
            sys.argv, sys.path = old_argv, old_path
            os.chdir(old_cwd)
            namespace.clear()
            gc.collect()


def main():
    runtime = ResidentScript(sys.argv[1])
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            break
        try:
            runtime.run(request["cmd"], request.get("cwd"))
            result = {"ok": True}
        except BaseException as exc:
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
