"""PyBullet renderer that draws the track and car from simulator states, with optional planner overlays."""

from __future__ import annotations

import math
import os

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from uncertain_racecar_gym.common import package_asset_path
from uncertain_racecar_gym.scenario import Scenario
from uncertain_racecar_gym.track import TrackModel


class PyBulletMirrorRenderer:
    """Off-screen PyBullet scene that mirrors the simulated car.

    render_mode is 'rgb_array_follow', 'rgb_array_birds_eye',
    'rgb_array_cinematic' (frames returned) or 'human' (GUI window, no
    frames). The car's pose is set directly from each state; PyBullet is used
    only for drawing, not for physics.
    """
    def __init__(self, scenario: Scenario, track: TrackModel, render_mode: str, width: int = 1280, height: int = 720):
        self.scenario = scenario
        self.track = track
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.client_id = None
        self.vehicle_id = None
        self.ghost_vehicle_id = None
        self._joint_names = {}
        self._ghost_joint_names = {}
        self._frame_index = 0
        self._camera_renderer = p.ER_BULLET_HARDWARE_OPENGL
        self._connect()
        self._build_scene()

    def _polyline_normals(self) -> np.ndarray:
        points = self.track.centerline
        previous = np.roll(points, 1, axis=0)
        nxt = np.roll(points, -1, axis=0)
        tangents = nxt - previous
        tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
        return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    def _create_strip_mesh(
        self,
        edge_a: np.ndarray,
        edge_b: np.ndarray,
        z_a: float,
        z_b: float,
        color: list[float],
        closed: bool = True,
    ) -> int:
        vertices = []
        indices = []
        point_count = len(edge_a)
        segment_count = point_count if closed else point_count - 1
        for index in range(point_count):
            vertices.append([float(edge_a[index, 0]), float(edge_a[index, 1]), float(z_a)])
            vertices.append([float(edge_b[index, 0]), float(edge_b[index, 1]), float(z_b)])
        for index in range(segment_count):
            nxt = (index + 1) % point_count
            base = 2 * index
            nxt_base = 2 * nxt
            indices.extend([base, base + 1, nxt_base + 1, base, nxt_base + 1, nxt_base])
        return p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=color,
            specularColor=[0.08, 0.08, 0.08],
            physicsClientId=self.client_id,
        )

    def _create_visual_body(self, visual_shape: int, position: list[float] | None = None) -> None:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=position or [0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

    def _connect(self) -> None:
        use_gui = self.render_mode == "human" and (os.environ.get("DISPLAY") or os.name == "nt" or os.uname().sysname == "Darwin")
        connection_mode = p.GUI if use_gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)

    def _build_scene(self) -> None:
        """Create the ground, track surface, kerbs, edge lines and centre dashes."""
        road_height = 0.02
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[120.0, 120.0, 0.02],
                physicsClientId=self.client_id,
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[120.0, 120.0, 0.02],
                rgbaColor=[0.2, 0.37, 0.2, 1.0],
                physicsClientId=self.client_id,
            ),
            basePosition=[0.0, 0.0, -0.02],
            physicsClientId=self.client_id,
        )

        normals = self._polyline_normals()
        centerline = self.track.centerline
        shoulder_outer = 0.5 * self.track.width + 1.3
        road_half_width = 0.5 * self.track.width
        stripe_outer = road_half_width - 0.04
        stripe_inner = road_half_width - 0.24

        left_shoulder_outer = centerline + normals * shoulder_outer
        right_shoulder_outer = centerline - normals * shoulder_outer
        left_edge = centerline + normals * road_half_width
        right_edge = centerline - normals * road_half_width
        left_stripe_inner = centerline + normals * stripe_inner
        right_stripe_inner = centerline - normals * stripe_inner

        shoulder_shape = self._create_strip_mesh(
            left_shoulder_outer,
            right_shoulder_outer,
            z_a=road_height * 0.96,
            z_b=road_height * 0.96,
            color=[0.22, 0.22, 0.24, 1.0],
        )
        self._create_visual_body(shoulder_shape)

        road_shape = self._create_strip_mesh(
            left_edge,
            right_edge,
            z_a=road_height,
            z_b=road_height,
            color=[0.12, 0.12, 0.14, 1.0],
        )
        self._create_visual_body(road_shape)

        left_edge_shape = self._create_strip_mesh(
            centerline + normals * stripe_outer,
            left_stripe_inner,
            z_a=road_height * 1.12,
            z_b=road_height * 1.12,
            color=[0.96, 0.96, 0.96, 1.0],
        )
        right_edge_shape = self._create_strip_mesh(
            right_stripe_inner,
            centerline - normals * stripe_outer,
            z_a=road_height * 1.12,
            z_b=road_height * 1.12,
            color=[0.96, 0.96, 0.96, 1.0],
        )
        self._create_visual_body(left_edge_shape)
        self._create_visual_body(right_edge_shape)

        curb_outer_width = road_half_width + 0.55
        curb_inner_width = road_half_width + 0.12
        left_curb_shape = self._create_strip_mesh(
            centerline + normals * curb_outer_width,
            centerline + normals * curb_inner_width,
            z_a=road_height * 1.02,
            z_b=road_height * 1.02,
            color=[0.78, 0.16, 0.16, 0.92],
        )
        right_curb_shape = self._create_strip_mesh(
            centerline - normals * curb_inner_width,
            centerline - normals * curb_outer_width,
            z_a=road_height * 1.02,
            z_b=road_height * 1.02,
            color=[0.78, 0.16, 0.16, 0.92],
        )
        self._create_visual_body(left_curb_shape)
        self._create_visual_body(right_curb_shape)

        guardrail_inner = centerline + normals * (road_half_width + 0.9)
        guardrail_outer = centerline + normals * (road_half_width + 1.02)
        guardrail_left = self._create_strip_mesh(
            guardrail_inner,
            guardrail_outer,
            z_a=0.30,
            z_b=0.34,
            color=[0.88, 0.9, 0.94, 0.92],
        )
        guardrail_right = self._create_strip_mesh(
            centerline - normals * (road_half_width + 1.02),
            centerline - normals * (road_half_width + 0.9),
            z_a=0.34,
            z_b=0.30,
            color=[0.88, 0.9, 0.94, 0.92],
        )
        self._create_visual_body(guardrail_left)
        self._create_visual_body(guardrail_right)

        center_dash_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.65, 0.045, 0.01],
            rgbaColor=[0.98, 0.86, 0.2, 0.95],
            physicsClientId=self.client_id,
        )
        dash_spacing = 4.8
        dash_length = 1.2
        dash_count = max(1, int(self.track.length / dash_spacing))
        for dash_index in range(dash_count):
            progress = (dash_index * dash_spacing) / max(self.track.length, 1e-9)
            start = self.track.sample(progress)
            end = self.track.sample(progress + dash_length / max(self.track.length, 1e-9))
            midpoint = np.array([(start.x + end.x) * 0.5, (start.y + end.y) * 0.5], dtype=float)
            yaw = math.atan2(end.y - start.y, end.x - start.x)
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=center_dash_shape,
                basePosition=[float(midpoint[0]), float(midpoint[1]), road_height * 1.2],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

        self.vehicle_id, self._joint_names = self._spawn_vehicle(rgba=None)

    def _spawn_vehicle(self, rgba: list[float] | None) -> tuple[int, dict[str, int]]:
        """Load the car URDF (assets/vehicles/simple_racecar.urdf)."""
        urdf_path = package_asset_path("vehicles/simple_racecar.urdf")
        vehicle_id = p.loadURDF(
            str(urdf_path),
            [0.0, 0.0, 0.2],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        joint_names: dict[str, int] = {}
        if rgba is not None:
            p.changeVisualShape(vehicle_id, -1, rgbaColor=rgba, physicsClientId=self.client_id)
        p.setCollisionFilterGroupMask(vehicle_id, -1, 0, 0, physicsClientId=self.client_id)
        for joint_index in range(p.getNumJoints(vehicle_id, physicsClientId=self.client_id)):
            joint_name = p.getJointInfo(vehicle_id, joint_index, physicsClientId=self.client_id)[1].decode("utf-8")
            joint_names[joint_name] = joint_index
            if rgba is not None:
                p.changeVisualShape(vehicle_id, joint_index, rgbaColor=rgba, physicsClientId=self.client_id)
            p.setCollisionFilterGroupMask(vehicle_id, joint_index, 0, 0, physicsClientId=self.client_id)
        return vehicle_id, joint_names

    def _ensure_ghost_vehicle(self) -> None:
        if self.ghost_vehicle_id is None:
            self.ghost_vehicle_id, self._ghost_joint_names = self._spawn_vehicle([0.32, 0.72, 1.0, 0.38])

    def _apply_vehicle_state(self, vehicle_id: int, joint_names: dict[str, int], render_state: dict) -> None:
        position = [render_state["x"], render_state["y"], 0.22]
        orientation = p.getQuaternionFromEuler([0.0, 0.0, render_state["yaw"]])
        p.resetBasePositionAndOrientation(vehicle_id, position, orientation, physicsClientId=self.client_id)

        steer_angle = render_state["steering_angle"]
        wheel_rotation = render_state["wheel_rotation"]
        front_steer_joints = ("front_left_steer", "front_right_steer")
        wheel_joints = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
        for name in front_steer_joints:
            p.resetJointState(vehicle_id, joint_names[name], targetValue=steer_angle, physicsClientId=self.client_id)
        for name in wheel_joints:
            p.resetJointState(vehicle_id, joint_names[name], targetValue=wheel_rotation, physicsClientId=self.client_id)
        self._frame_index = int(render_state.get("frame_index", self._frame_index + 1))

    def update(self, render_state: dict, comparison_state: dict | None = None) -> None:
        """Place the car (and an optional second 'ghost' car) at the given states."""
        self._apply_vehicle_state(self.vehicle_id, self._joint_names, render_state)
        if comparison_state is not None:
            self._ensure_ghost_vehicle()
            self._apply_vehicle_state(self.ghost_vehicle_id, self._ghost_joint_names, comparison_state)
        p.stepSimulation(physicsClientId=self.client_id)

    def _camera(self, render_state: dict, mode: str):
        """View and projection matrices for the chosen camera mode."""
        x = render_state["x"]
        y = render_state["y"]
        yaw = render_state["yaw"]

        if mode == "birds_eye":
            target = [x, y, 0.0]
            view = p.computeViewMatrixFromYawPitchRoll(target, distance=18.0, yaw=0.0, pitch=-89.0, roll=0.0, upAxisIndex=2)
        elif mode == "cinematic":
            orbit_yaw = math.degrees((self._frame_index * 0.03) % (2.0 * math.pi))
            target = [x, y, 0.0]
            view = p.computeViewMatrixFromYawPitchRoll(target, distance=7.5, yaw=orbit_yaw, pitch=-20.0, roll=0.0, upAxisIndex=2)
        else:
            camera_position = np.array([x, y, 0.45]) + np.array([-4.5 * math.cos(yaw), -4.5 * math.sin(yaw), 1.6])
            target = np.array([x, y, 0.35]) + np.array([2.5 * math.cos(yaw), 2.5 * math.sin(yaw), 0.2])
            view = p.computeViewMatrix(camera_position, target, [0.0, 0.0, 1.0])

        projection = p.computeProjectionMatrixFOV(fov=60.0, aspect=float(self.width) / float(self.height), nearVal=0.05, farVal=100.0)
        return view, projection

    @staticmethod
    def _matrix4(values: list[float] | tuple[float, ...]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape((4, 4), order="F")

    def _project_xy(self, xy_points: np.ndarray, view: list[float] | tuple[float, ...], projection: list[float] | tuple[float, ...]) -> np.ndarray:
        """Project world x-y points to image pixels, for drawing overlays."""
        if xy_points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        view_matrix = self._matrix4(view)
        projection_matrix = self._matrix4(projection)
        points = np.asarray(xy_points, dtype=np.float32)
        homogeneous = np.concatenate(
            [
                points,
                np.full((points.shape[0], 1), 0.28, dtype=np.float32),
                np.ones((points.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        clip = (projection_matrix @ view_matrix @ homogeneous.T).T
        w = np.maximum(clip[:, 3:4], 1e-6)
        ndc = clip[:, :3] / w
        pixels = np.empty((points.shape[0], 2), dtype=np.float32)
        pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * float(self.width - 1)
        pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(self.height - 1)
        valid = np.logical_and.reduce(
            [
                clip[:, 3] > 1e-6,
                ndc[:, 2] > -1.5,
                ndc[:, 2] < 1.5,
            ]
        )
        pixels[~valid] = np.nan
        return pixels

    def _overlay_planner_debug(
        self,
        frame: np.ndarray,
        render_state: dict,
        planner_debug: dict | None,
        view: list[float] | tuple[float, ...],
        projection: list[float] | tuple[float, ...],
    ) -> np.ndarray:
        """Draw planner information on a frame.

        Without gatekeeper data it draws the MPPI candidate and final paths. With
        gatekeeper data it draws the rollouts (nominal part green, backup part
        blue, or red if that rollout failed), the switching point, and a HUD with
        the active policy, step, switching time, speed, failure bound and
        controls.
        """
        if planner_debug is None:
            return frame
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        candidate_xy = np.asarray(planner_debug.get("candidate_xy", np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        final_xy = np.asarray(planner_debug.get("final_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        
        # MPPI overlay (only when there are no gatekeeper rollouts).
        gk_trajectories = planner_debug.get("gk_trajectories") # [N, T+1, 2]
        
        if gk_trajectories is None:
            for trajectory in candidate_xy:
                pixels = self._project_xy(trajectory, view, projection)
                pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
                pixels = pixels[2:]
                if len(pixels) >= 2:
                    draw.line([tuple(point) for point in pixels], fill=(84, 180, 255, 46), width=1)

            if len(final_xy) >= 2:
                pixels = self._project_xy(final_xy, view, projection)
                pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
                pixels = pixels[2:]
                if len(pixels) >= 2:
                    draw.line([tuple(point) for point in pixels], fill=(255, 184, 44, 255), width=4)
                    draw.line([tuple(point) for point in pixels], fill=(255, 236, 180, 176), width=2)
        
        # Gatekeeper overlay.
        failure_mask = planner_debug.get("failure_mask") # [N] bool
        m_star = planner_debug.get("m_star", 0)
        using_backup = planner_debug.get("using_backup", False)
        q_bar_star = planner_debug.get("q_bar_star", 0.0)
        speed = render_state.get("vx", 0.0)

        if gk_trajectories is not None:
            # Plan pinning logic:
            # We must index the rollout relative to when it was first calculated.
            plan_start_t = planner_debug.get("plan_start_t", 0)
            s_t = planner_debug.get("s_t", 0)
            local_m = s_t - plan_start_t
            
            # For slicing, we still want to clamp to the visible part [0, T_window]
            m_slice = max(0, min(local_m, len(gk_trajectories[0]) - 1))
            
            for i, traj in enumerate(gk_trajectories):
                pixels = self._project_xy(traj, view, projection)
                valid_mask = np.all(np.isfinite(pixels), axis=1)
                
                # Draw Nominal (Green)
                nom_pixels = pixels[1:m_slice + 2][valid_mask[1:m_slice + 2]]
                if len(nom_pixels) >= 2:
                    draw.line([tuple(p) for p in nom_pixels], fill=(34, 193, 114, 60), width=1)
                
                # Draw Backup (Red if failed, Blue if safe)
                is_failed = False
                if failure_mask is not None and i < len(failure_mask):
                    is_failed = bool(failure_mask[i])
                
                backup_color = (255, 30, 30, 80) if is_failed else (52, 152, 219, 60)
                
                back_pixels = pixels[m_slice + 2:][valid_mask[m_slice + 2:]]
                if len(back_pixels) >= 2:
                    draw.line([tuple(p) for p in back_pixels], fill=backup_color, width=1)
                
                # Draw Switching Point (Yellow Dot) at the actual local index
                if local_m +1 < len(pixels) and local_m +1 >= 0 and valid_mask[local_m+1]:
                    px, py = pixels[local_m+1]
                    draw.ellipse([px-2, py-2, px+2, py+2], fill=(255, 235, 59, 200))

        # --- Text Info Overlay ---
        from PIL import ImageFont
        try:
            # Try to load a nice font, fallback to default
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Dashboard background
        draw.rectangle([10, 10, 260, 145], fill=(0, 0, 0, 160))
        
        # Policy Status
        is_reverting = planner_debug.get("is_reverting", False)
        policy_text = "BACKUP" if using_backup else "NOMINAL"
        policy_color = (255, 87, 34, 255) if using_backup else (76, 175, 80, 255)
        draw.text((20, 20), policy_text, fill=policy_color, font=font)
        
        if is_reverting:
            draw.text((130, 20), "REVERT", fill=(255, 235, 59, 255), font=font)
        
        # Step Count
        t = render_state.get("frame_index", 0)
        max_t = render_state.get("total_frames")
        step_text = f"Step: {t}"
        if max_t:
            step_text += f"/{max_t}"
        draw.text((20, 48), step_text, fill=(255, 255, 255, 255), font=small_font)
        
        # Switching Time (Absolute and Relative)
        s_t = planner_debug.get("s_t")
        if s_t is not None:
            m_star = planner_debug.get("m_star", 0)
            st_text = f"Switch: {s_t} (T{m_star:+d})"
            draw.text((20, 65), st_text, fill=(255, 255, 255, 255), font=small_font)
        
        # Speed
        draw.text((20, 85), f"Speed: {speed:.1f} m/s", fill=(255, 255, 255, 255), font=small_font)
        
        # Failure Prob
        epsilon = planner_debug.get("epsilon", 0.1)
        prob_color = (255, 255, 255, 255)
        if q_bar_star > 0.5 * epsilon: 
            prob_color = (255, 235, 59, 255) # Yellow
        if q_bar_star > 0.8 * epsilon:
            prob_color = (255, 152, 0, 255)  # Orange
        if q_bar_star > 1.0 * epsilon:
            prob_color = (255, 30, 30, 255)  # Red
             
        draw.text((20, 105), f"P(fail) Bound: {q_bar_star:.3f}", fill=prob_color, font=small_font)
        draw.text((20, 120), f"(Threshold eps: {epsilon:.2f})", fill=(180, 180, 180, 255), font=small_font)

        # --- Control Inputs Graphic ---
        action = planner_debug.get("action")
        if action is not None and len(action) >= 3:
            steer, throttle, brake = action[0], action[1], action[2]
            
            # Anchor for control HUD (moved down to avoid overlap)
            hud_y_start = 120
            
            # Semi-transparent background for controls
            # Increased height for vertical bars
            draw.rectangle([10, hud_y_start, 260, hud_y_start + 130], fill=(0, 0, 0, 140))
            
            # 1. Steering Slider (Horizontal)
            # Box: [20, hud_y_start + 10, 250, hud_y_start + 25]
            steer_box = [25, hud_y_start + 15, 245, hud_y_start + 30]
            draw.rectangle(steer_box, outline=(200, 200, 200, 200), width=1)
            # Center line
            draw.line([135, steer_box[1], 135, steer_box[3]], fill=(150, 150, 150, 255), width=1)
            # Steering Indicator
            indicator_x = 135 + (steer * 105) # Center at 135, half-width 105
            draw.rectangle([indicator_x - 3, steer_box[1], indicator_x + 3, steer_box[3]], fill=(3, 169, 244, 255))
            draw.text((25, hud_y_start + 1), "STEERING", fill=(200, 200, 200, 180), font=small_font)
            
            # 2. Throttle and Brake (Vertical Bars)
            v_bar_h = 70
            v_bar_w = 40
            v_bar_y = hud_y_start + 50
            
            # Throttle (Vertical)
            t_box = [60, v_bar_y, 60 + v_bar_w, v_bar_y + v_bar_h]
            draw.rectangle(t_box, outline=(200, 200, 200, 200), width=1)
            t_fill_h = max(0, min(v_bar_h - 2, int(throttle * (v_bar_h - 2))))
            draw.rectangle([t_box[0] + 1, t_box[3] - 1 - t_fill_h, t_box[2] - 1, t_box[3] - 1], fill=(76, 175, 80, 255))
            draw.text((t_box[0], v_bar_y - 18), "THR", fill=(76, 175, 80, 180), font=small_font)
            
            # Brake (Vertical)
            b_box = [170, v_bar_y, 170 + v_bar_w, v_bar_y + v_bar_h]
            draw.rectangle(b_box, outline=(200, 200, 200, 200), width=1)
            b_fill_h = max(0, min(v_bar_h - 2, int(brake * (v_bar_h - 2))))
            draw.rectangle([b_box[0] + 1, b_box[3] - 1 - b_fill_h, b_box[2] - 1, b_box[3] - 1], fill=(244, 67, 54, 255))
            draw.text((b_box[0], v_bar_y - 18), "BRK", fill=(244, 67, 54, 180), font=small_font)

        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def render(
        self,
        render_state: dict,
        comparison_state: dict | None = None,
        planner_debug: dict | None = None,
    ) -> np.ndarray | None:
        """Draw one frame for `render_state` and return it as an RGB array.

        render_state needs x, y, yaw, steering_angle and wheel_rotation;
        planner_debug adds the overlays. Returns None in 'human' mode.
        """
        mode = self.render_mode.replace("rgb_array_", "")
        if self.render_mode == "human" and p.getConnectionInfo(self.client_id)["connectionMethod"] == p.GUI:
            self.update(render_state, comparison_state=comparison_state)
            return None

        self.update(render_state, comparison_state=comparison_state)
        view, projection = self._camera(render_state, mode if mode in {"follow", "birds_eye", "cinematic"} else "follow")
        try:
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                renderer=self._camera_renderer,
                viewMatrix=view,
                projectionMatrix=projection,
                shadow=1,
                lightDirection=[1.2, -0.8, 2.6],
                lightColor=[1.0, 0.98, 0.95],
                lightAmbientCoeff=0.65,
                lightDiffuseCoeff=0.55,
                lightSpecularCoeff=0.15,
                physicsClientId=self.client_id,
            )
        except Exception:
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                renderer=p.ER_TINY_RENDERER,
                viewMatrix=view,
                projectionMatrix=projection,
                physicsClientId=self.client_id,
            )
        frame = np.reshape(rgb, (self.height, self.width, 4))[:, :, :3]
        return self._overlay_planner_debug(frame, render_state, planner_debug, view, projection)

    def close(self) -> None:
        """Disconnect from PyBullet."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None


