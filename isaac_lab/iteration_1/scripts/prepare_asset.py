"""Export the saved Iteration_1 robot and static optical calibration into this project."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from pxr import Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/zalewm3/nursebot_sim/ai_bridge/scenes/Iteration_1.usd')

def pose(matrix):
    q = matrix.ExtractRotationQuat()
    return dict(pos=list(matrix.ExtractTranslation()), quat_xyzw=list(q.GetImaginary())+[q.GetReal()])

def main():
    output = ROOT/'assets/oarbot_iteration_1.usdc'
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        shutil.copy2(output, output.with_suffix('.before_refresh.usdc'))
        output.unlink()
    source = Usd.Stage.Open(str(SOURCE))
    if not source.GetPrimAtPath('/oarbot'):
        raise RuntimeError('Saved scene does not contain /oarbot')
    flat = source.Flatten()
    target_layer = Sdf.Layer.CreateNew(str(output))
    if not Sdf.CopySpec(flat, '/oarbot', target_layer, '/oarbot'):
        raise RuntimeError('Robot export failed')
    target = Usd.Stage.Open(target_layer)
    target.SetDefaultPrim(target.GetPrimAtPath('/oarbot'))
    UsdGeom.SetStageMetersPerUnit(target, 1.0)
    UsdGeom.SetStageUpAxis(target, UsdGeom.Tokens.z)
    # Copy material targets into the referenced robot subtree and remap shader connections.
    material_map = {}
    UsdGeom.Scope.Define(target, '/oarbot/TrainingMaterials')
    for prim in list(target.Traverse()):
        for rel in prim.GetRelationships():
            if rel.GetName() == 'physics:simulationOwner':
                rel.ClearTargets(True)
                continue
            replacements = []
            for path in rel.GetTargets():
                if path.HasPrefix(Sdf.Path('/oarbot')):
                    replacements.append(path)
                    continue
                original = source.GetPrimAtPath(path.GetPrimPath())
                material = original
                while material and material.GetTypeName() != 'Material':
                    material = material.GetParent()
                if not material:
                    replacements.append(path)
                    continue
                old = material.GetPath()
                if old not in material_map:
                    new = Sdf.Path(f'/oarbot/TrainingMaterials/Material{len(material_map)}')
                    Sdf.CopySpec(flat, old, target_layer, new)
                    material_map[old] = new
                replacements.append(path.ReplacePrefix(old, material_map[old]))
            rel.SetTargets(replacements)
    for old, new in material_map.items():
        for prim in Usd.PrimRange(target.GetPrimAtPath(new)):
            for attr in prim.GetAttributes():
                connections = attr.GetConnections()
                if connections:
                    attr.SetConnections([p.ReplacePrefix(old, new) if p.HasPrefix(old) else p for p in connections])
            for rel in prim.GetRelationships():
                rel.SetTargets([p.ReplacePrefix(old, new) if p.HasPrefix(old) else p for p in rel.GetTargets()])
    unresolved = []
    for prim in target.Traverse():
        for rel in prim.GetRelationships():
            for path in rel.GetTargets():
                if path.IsAbsolutePath() and not target.GetPrimAtPath(path.GetPrimPath()):
                    unresolved.append([str(prim.GetPath()), rel.GetName(), str(path)])
    if unresolved:
        raise RuntimeError(f'Export contains unresolved relationships: {unresolved[:8]}')
    target.GetRootLayer().Save()

    calibration = {}
    xforms = UsdGeom.XformCache()
    for device, rgb_path in (
        ('room', '/OarBotAI/CornerCamera'),
        ('wrist', '/oarbot/j2n6s300_link_6/ArmCamera'),
    ):
        rgb_prim = source.GetPrimAtPath(rgb_path)
        depth_prim = source.GetPrimAtPath(rgb_path+'/Depth')
        parent = source.GetPrimAtPath('/oarbot/j2n6s300_link_6') if device == 'wrist' else source.GetPseudoRoot()
        entry = {}
        for channel, prim in (('rgb', rgb_prim), ('depth', depth_prim)):
            if not prim.IsA(UsdGeom.Camera): raise RuntimeError(f'Missing {device} {channel} camera')
            if device == 'wrist':
                transform = xforms.GetLocalToWorldTransform(prim)*xforms.GetLocalToWorldTransform(parent).GetInverse()
            else:
                transform = xforms.GetLocalToWorldTransform(prim)
            camera = UsdGeom.Camera(prim)
            entry[channel] = dict(**pose(transform), focal=float(camera.GetFocalLengthAttr().Get()),
                h_ap=float(camera.GetHorizontalApertureAttr().Get()),
                v_ap=float(camera.GetVerticalApertureAttr().Get()))
        depth_to_rgb = xforms.GetLocalToWorldTransform(depth_prim)*xforms.GetLocalToWorldTransform(rgb_prim).GetInverse()
        entry['depth_to_rgb'] = pose(depth_to_rgb)
        calibration[device] = entry
    calibration['model_note'] = 'Ideal pinholes; source mount geometry and separate RGB/depth frusta. Factory lens distortion and measured hardware extrinsics are not calibrated.'
    (ROOT/'assets/sensor_mounts.json').write_text(json.dumps(calibration, indent=2)+'\n')
    report = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        asset=str(output), asset_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(), up_axis='Z', meters_per_unit=1.0,
        materials_copied=len(material_map), unresolved_relationships=unresolved,
        robot_geometry_and_joint_properties_changed=False, sensor_calibration=calibration)
    folder=ROOT/'verification'; folder.mkdir(exist_ok=True)
    (folder/'asset_export.json').write_text(json.dumps(report, indent=2)+'\n')
    print('ASSET_EXPORT_PASS', json.dumps({k:report[k] for k in ('source','asset','materials_copied','up_axis','meters_per_unit')}))

if __name__ == '__main__': main()
