"""Viewer framing and camera icon visibility; never changes a sensor's measurements."""
import math

def configure_view(cfg):
    if cfg.scene.num_envs > 1:
        span = math.ceil(math.sqrt(cfg.scene.num_envs))*cfg.scene.env_spacing
        cfg.viewer.eye = (.8*span, -.8*span, .8*span)
        cfg.viewer.lookat = (1.25, 0., .5)

def hide_camera_icons():
    try:
        import omni.kit.actions.core
        action = omni.kit.actions.core.get_action_registry().get_action(
            'omni.kit.viewport.actions', 'toggle_camera_visibility')
        if action:
            action.execute(visible=False)
    except (ImportError, AttributeError):
        pass
