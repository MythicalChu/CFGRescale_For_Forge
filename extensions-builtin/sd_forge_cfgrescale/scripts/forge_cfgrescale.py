import gradio as gr

from modules import scripts
from ldm_patched.contrib.external_cfgrescale import RescaleCFG

opRescaleCFG = RescaleCFG()

class CFGRescaleForForge(scripts.Script):
    def title(self):
        return "CFGRescale Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            cfgrescale_enabled = gr.Checkbox(label='Enabled', value=True)
            multiplier = gr.Slider(label='Multiplier', minimum=0, maximum=1, step=0.01, value=0.7)

        return cfgrescale_enabled, multiplier

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        cfgrescale_enabled, multiplier = script_args

        if not cfgrescale_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opRescaleCFG.patch(unet, multiplier)[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            cfgrescale_enabled=cfgrescale_enabled,
            multiplier=multiplier,
        ))

        return