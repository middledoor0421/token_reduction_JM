# methods/baselines/identity/plugin.py
# Identity (no reduction) plugin, conforms to TokenReducerPlugin API.

from core.registry import TokenReducerPlugin, register

@register("identity")
class IdentityPlugin(TokenReducerPlugin):
    def attach(self, model):
        # No modification to the model. Acts as a pass-through.
        self.model = model
        print("[plugin/identity] attached (no reduction).")

    def finalize(self):
        print("[plugin/identity] done.")
