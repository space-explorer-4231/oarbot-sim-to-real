"""Static authored wrist-to-tool transform; used only for training scores."""
import math
from isaaclab.utils.math import combine_frame_transforms

def tool_pose(robot, wrist_id):
    wrist = robot.data.body_link_pose_w.torch[:, wrist_id]
    offset = wrist.new_tensor((0., 0., -0.16)).expand_as(wrist[:, :3])
    rotation = wrist.new_tensor((math.sqrt(.5), math.sqrt(.5), 0., 0.)).expand_as(wrist[:, 3:])
    return combine_frame_transforms(wrist[:, :3], wrist[:, 3:], offset, rotation)
