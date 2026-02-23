"""

F1 Track Neural Network Racing Simulation

OPTIMIZED VERSION - Pre-rendered track, spatial grid, vectorized sensors

"""


import pygame
import numpy as np
import math
import random
import os
import torch


# Better CUDA detection

def setup_device():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

device = setup_device()



WIDTH, HEIGHT = 1000, 600
POPULATION = 100
SENSOR_COUNT = 8
LOOKAHEAD_COUNT = 3
MAX_SENSOR_RANGE = 100
MAX_SPEED = 30
BASE_MUTATION_RATE = 0.1
GENERATION_TIME = 30.0
CHECKPOINT_COUNT = 60
TARGET_FPS = 60
STAGNATION_GENS = 8
IDLE_TIMEOUT = 5.0
TOTAL_INPUTS = SENSOR_COUNT + LOOKAHEAD_COUNT * 2 + 3
HIDDEN_SIZE = 32

CRASH_MEMORY_RADIUS = 80
GRID_CELL_SIZE = 100  # For spatial partitioning
SENSOR_OFFSETS = np.linspace(-1.0, 1.0, SENSOR_COUNT, dtype=np.float32)

# ---------------------------------------


os.environ['SDL_VIDEO_CENTERED'] = '1'

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)

pygame.display.set_caption("F1 Neural Racing - Fast")

clock = pygame.time.Clock()


font = pygame.font.SysFont("Consolas", 15)

font_sm = pygame.font.SysFont("Consolas", 13)

font_large = pygame.font.SysFont("Consolas", 22, bold=True)



class Camera:

    __slots__ = ['x', 'y', 'zoom', 'target_x', 'target_y', 'target_zoom']


    def __init__(self):

        self.x = self.y = 0.0

        self.zoom = 0.5

        self.target_x = self.target_y = 0.0

        self.target_zoom = 0.5


    def follow(self, x, y):

        self.target_x, self.target_y = x, y


    def update(self, dt):

        t = min(1.0, dt * 6.0)

        self.x += (self.target_x - self.x) * t

        self.y += (self.target_y - self.y) * t

        self.zoom += (self.target_zoom - self.zoom) * t


    def zoom_in(self):

        self.target_zoom = min(3.0, self.target_zoom * 1.3)


    def zoom_out(self):

        self.target_zoom = max(0.05, self.target_zoom / 1.3)

   

    def world_to_screen(self, wx, wy):

        return ((wx - self.x) * self.zoom + WIDTH * 0.5,

                (wy - self.y) * self.zoom + HEIGHT * 0.5)



class Track:

    def __init__(self, track_name):

        self.name = track_name

        self._load_track_csv(track_name)

        self._compute_boundaries()

        self._build_spatial_grid()

        self._generate_checkpoints()

        self._compute_spawn()

        self.crash_points = []

        self.crash_heatmap = {}

        # Pre-rendered track surface

        self._cached_surface = None

        self._cached_cam_state = None


    def _load_track_csv(self, name):

        base = os.path.dirname(__file__)

        path = os.path.join(base, "tracks", f"{name}.csv")

       

        cx, cy, wr, wl = [], [], [], []

        with open(path, "r") as f:

            for line in f:

                line = line.strip()

                if not line or line.startswith("#"):

                    continue

                parts = line.split(",")

                cx.append(float(parts[0]))

                cy.append(-float(parts[1]))

                wr.append(float(parts[2]) if len(parts) > 2 else 6.0)

                wl.append(float(parts[3]) if len(parts) > 3 else 6.0)

       

        cx, cy = np.array(cx), np.array(cy)

        cx -= (cx.min() + cx.max()) / 2

        cy -= (cy.min() + cy.max()) / 2

       

        self.center_x, self.center_y = cx, cy

        self.w_right = np.array(wr) * 5

        self.w_left = np.array(wl) * 5

        self.n = len(cx)

       

        # Precompute bounds

        self.min_x, self.max_x = cx.min() - 100, cx.max() + 100

        self.min_y, self.max_y = cy.min() - 100, cy.max() + 100


    def _compute_boundaries(self):

        n = self.n
        # Vectorized normal computation (no Python loop)
        next_i = (np.arange(n) + 1) % n

        prev_i = (np.arange(n) - 1) % n

        dx = self.center_x[next_i] - self.center_x[prev_i]

        dy = self.center_y[next_i] - self.center_y[prev_i]

        length = np.sqrt(dx * dx + dy * dy) + 1e-9

        nx = -dy / length

        ny = dx / length


        self.left_x = self.center_x + nx * self.w_left

        self.left_y = self.center_y + ny * self.w_left

        self.right_x = self.center_x - nx * self.w_right

        self.right_y = self.center_y - ny * self.w_right


        # Build segment array with vectorized indexing

        ni = (np.arange(n) + 1) % n

        self.seg_array = np.zeros((n * 2, 4), dtype=np.float32)

        self.seg_array[0::2, 0] = self.left_x

        self.seg_array[0::2, 1] = self.left_y

        self.seg_array[0::2, 2] = self.left_x[ni]

        self.seg_array[0::2, 3] = self.left_y[ni]

        self.seg_array[1::2, 0] = self.right_x

        self.seg_array[1::2, 1] = self.right_y

        self.seg_array[1::2, 2] = self.right_x[ni]

        self.seg_array[1::2, 3] = self.right_y[ni]


    def _build_spatial_grid(self):

        """Build spatial hash grid for fast segment lookup."""

        self.grid = {}

        for idx in range(len(self.seg_array)):

            seg = self.seg_array[idx]

            min_gx = int(min(seg[0], seg[2]) // GRID_CELL_SIZE)

            max_gx = int(max(seg[0], seg[2]) // GRID_CELL_SIZE)

            min_gy = int(min(seg[1], seg[3]) // GRID_CELL_SIZE)

            max_gy = int(max(seg[1], seg[3]) // GRID_CELL_SIZE)

            for gx in range(min_gx, max_gx + 1):

                for gy in range(min_gy, max_gy + 1):

                    key = (gx, gy)

                    if key not in self.grid:

                        self.grid[key] = []

                    self.grid[key].append(idx)


    def get_nearby_segments(self, x, y, radius):

        """Get segment indices near a point."""

        gx, gy = int(x // GRID_CELL_SIZE), int(y // GRID_CELL_SIZE)

        r = int(radius / GRID_CELL_SIZE) + 1

        indices = set()

        for dx in range(-r, r + 1):

            for dy in range(-r, r + 1):

                key = (gx + dx, gy + dy)

                if key in self.grid:

                    indices.update(self.grid[key])

        return indices


    def _generate_checkpoints(self):

        self.checkpoints = []

        self.cp_centers = []

        for k in range(CHECKPOINT_COUNT):

            idx = int((k / CHECKPOINT_COUNT) * self.n) % self.n

            cp = (self.left_x[idx], self.left_y[idx], self.right_x[idx], self.right_y[idx])

            self.checkpoints.append(cp)

            self.cp_centers.append(((cp[0] + cp[2]) / 2, (cp[1] + cp[3]) / 2))

        self.cp_centers_np = np.array(self.cp_centers, dtype=np.float32)


    def _compute_spawn(self):

        cp = self.checkpoints[0]

        self.spawn_x = (cp[0] + cp[2]) / 2

        self.spawn_y = (cp[1] + cp[3]) / 2

        next_idx = int((1 / CHECKPOINT_COUNT) * self.n) % self.n

        dx = self.center_x[next_idx] - self.center_x[0]

        dy = self.center_y[next_idx] - self.center_y[0]

        self.spawn_angle = math.atan2(dy, dx)


    def is_on_track(self, x, y):

        dx = self.center_x - x

        dy = self.center_y - y

        dists = dx * dx + dy * dy

        idx = np.argmin(dists)

        max_w = (self.w_left[idx] + self.w_right[idx]) / 2 + 2

        return dists[idx] < max_w * max_w


    def batch_is_on_track(self, xs, ys):
        """Vectorized track check for multiple cars at once."""
        dx = self.center_x[None, :] - xs[:, None]
        dy = self.center_y[None, :] - ys[:, None]
        dists = dx * dx + dy * dy
        idx = np.argmin(dists, axis=1)
        min_d = dists[np.arange(len(xs)), idx]
        max_w = (self.w_left[idx] + self.w_right[idx]) / 2 + 2
        return min_d < max_w * max_w


    def draw(self, camera, next_cp=None):

        """Draw track - use cached surface when possible."""

        cam_state = (int(camera.x), int(camera.y), round(camera.zoom, 2))

       

        if self._cached_surface is None or self._cached_cam_state != cam_state:

            self._render_track_surface(camera)

            self._cached_cam_state = cam_state

       

        screen.blit(self._cached_surface, (0, 0))

       

        # Only draw active checkpoint (not all)

        if next_cp is not None:

            cp = self.checkpoints[next_cp]

            a = camera.world_to_screen(cp[0], cp[1])

            b = camera.world_to_screen(cp[2], cp[3])

            pygame.draw.line(screen, (0, 255, 200), a, b, 3)


    def _render_track_surface(self, camera):

        """Render track to cached surface."""

        self._cached_surface = pygame.Surface((WIDTH, HEIGHT))

        self._cached_surface.fill((35, 65, 45))


        n = self.n

        # Vectorized world-to-screen for all points

        lx_s = (self.left_x - camera.x) * camera.zoom + WIDTH * 0.5

        ly_s = (self.left_y - camera.y) * camera.zoom + HEIGHT * 0.5

        rx_s = (self.right_x - camera.x) * camera.zoom + WIDTH * 0.5

        ry_s = (self.right_y - camera.y) * camera.zoom + HEIGHT * 0.5


        left_pts = list(zip(lx_s.tolist(), ly_s.tolist()))

        right_pts = list(zip(rx_s.tolist(), ry_s.tolist()))


        # Single polygon: left boundary forward + right boundary reversed

        track_polygon = left_pts + right_pts[::-1]

        if len(track_polygon) >= 3:

            pygame.draw.polygon(self._cached_surface, (50, 50, 55), track_polygon)


        # Clean boundary lines using ALL points (no skipping)

        if n > 2:

            pygame.draw.lines(self._cached_surface, (200, 200, 200), True, left_pts, 2)

            pygame.draw.lines(self._cached_surface, (200, 200, 200), True, right_pts, 2)


        # Dashed center line

        cx_s = (self.center_x - camera.x) * camera.zoom + WIDTH * 0.5

        cy_s = (self.center_y - camera.y) * camera.zoom + HEIGHT * 0.5

        for i in range(0, n - 1, 4):

            pygame.draw.line(self._cached_surface, (65, 65, 70),

                             (cx_s[i], cy_s[i]), (cx_s[i + 1], cy_s[i + 1]), 1)


    def record_crash(self, x, y):

        grid_x, grid_y = int(x / 50), int(y / 50)

        key = (grid_x, grid_y)

        self.crash_heatmap[key] = self.crash_heatmap.get(key, 0) + 1

        if self.crash_heatmap[key] >= 3:

            for i, (px, py, count) in enumerate(self.crash_points):

                if (px - x) ** 2 + (py - y) ** 2 < CRASH_MEMORY_RADIUS ** 2:

                    self.crash_points[i] = (px, py, count + 1)

                    return

            self.crash_points.append((x, y, 1))

   

    def get_danger_level(self, x, y):

        if not self.crash_points:

            return 0.0

        max_danger = 0.0

        for px, py, count in self.crash_points:

            d2 = (px - x) ** 2 + (py - y) ** 2

            if d2 < CRASH_MEMORY_RADIUS ** 2:

                dist = math.sqrt(d2)

                danger = (1.0 - dist / CRASH_MEMORY_RADIUS) * min(1.0, count / 5.0)

                if danger > max_danger:

                    max_danger = danger

        return max_danger



class NeuralNet:

    __slots__ = ['w1', 'b1', 'w2', 'b2']

   

    def __init__(self):

        self.w1 = torch.randn(TOTAL_INPUTS, HIDDEN_SIZE, device=device) * 0.5

        self.b1 = torch.zeros(HIDDEN_SIZE, device=device)

        self.w2 = torch.randn(HIDDEN_SIZE, 3, device=device) * 0.5

        self.b2 = torch.zeros(3, device=device)


    def mutate(self, rate=BASE_MUTATION_RATE):

        with torch.no_grad():

            self.w1 += torch.randn_like(self.w1) * rate * (torch.rand_like(self.w1) < 0.3)

            self.w2 += torch.randn_like(self.w2) * rate * (torch.rand_like(self.w2) < 0.3)

            self.b1 += torch.randn_like(self.b1) * rate * (torch.rand_like(self.b1) < 0.3)

            self.b2 += torch.randn_like(self.b2) * rate * (torch.rand_like(self.b2) < 0.3)


    def copy(self):

        new = NeuralNet.__new__(NeuralNet)

        new.w1, new.w2 = self.w1.clone(), self.w2.clone()

        new.b1, new.b2 = self.b1.clone(), self.b2.clone()

        return new



class BatchedBrains:

    """Persistent batched brain structure - reuse across frames."""

    def __init__(self, brains):

        self.n = len(brains)

        self.w1 = torch.stack([b.w1 for b in brains])

        self.b1 = torch.stack([b.b1 for b in brains])

        self.w2 = torch.stack([b.w2 for b in brains])

        self.b2 = torch.stack([b.b2 for b in brains])

        # Pre-allocate input buffer on device

        self._input_buf = torch.zeros((self.n, TOTAL_INPUTS), dtype=torch.float32, device=device)


    @torch.no_grad()

    def forward(self, inputs_np, alive_mask):

        """

        Batched inference - only processes alive cars via index gather.

        inputs_np: (N, TOTAL_INPUTS) numpy array

        alive_mask: (N,) bool numpy array

        Returns: (N, 3) numpy array

        """

        alive_idx = np.where(alive_mask)[0]

        if len(alive_idx) == 0:

            return np.zeros((self.n, 3), dtype=np.float32)


        n_alive = len(alive_idx)

        # Transfer only alive inputs to GPU

        self._input_buf[:n_alive] = torch.from_numpy(

            inputs_np[alive_idx]

        ).to(device, non_blocking=True)

        x = self._input_buf[:n_alive]


        # Gather weights for alive cars only

        w1 = self.w1[alive_idx]

        b1 = self.b1[alive_idx]

        w2 = self.w2[alive_idx]

        b2 = self.b2[alive_idx]


        # Forward pass

        h = torch.tanh(torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1)

        out = torch.tanh(torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2)


        result = np.zeros((self.n, 3), dtype=np.float32)

        result[alive_idx] = out.cpu().numpy()

        return result



# Fully vectorized ray-segment intersection (numpy broadcasting)

def batch_ray_intersect(ox, oy, angles, seg_array, max_range):

    """

    Cast N rays from (ox, oy) against M segments using numpy broadcasting.

    No Python loops - entire computation is vectorized.


    seg_array: (M, 4) numpy array of [ax, ay, bx, by]

    Returns: (N,) normalized distances in [0, 1]

    """

    n_rays = len(angles)

    if len(seg_array) == 0:

        return np.ones(n_rays, dtype=np.float32)


    cos_a = np.cos(angles).astype(np.float32)          # (N,)

    sin_a = np.sin(angles).astype(np.float32)          # (N,)

    seg = np.asarray(seg_array, dtype=np.float32)      # (M, 4)


    sx = seg[:, 2] - seg[:, 0]                         # (M,) segment dx

    sy = seg[:, 3] - seg[:, 1]                         # (M,) segment dy

    dox = seg[:, 0] - ox                               # (M,) origin-to-seg x

    doy = seg[:, 1] - oy                               # (M,) origin-to-seg y


    # (N, M) via broadcasting

    denom = cos_a[:, None] * sy[None, :] - sin_a[:, None] * sx[None, :]

    t_num = dox * sy - doy * sx                        # (M,) ray-independent

    u_num = dox[None, :] * sin_a[:, None] - doy[None, :] * cos_a[:, None]


    abs_denom = np.abs(denom)

    safe_denom = np.where(abs_denom > 1e-9, denom, 1.0)

    t = t_num[None, :] / safe_denom                    # (N, M)

    u = u_num / safe_denom                             # (N, M)


    valid = (abs_denom > 1e-9) & (t > 0) & (t < max_range) & (u >= 0) & (u <= 1)

    t = np.where(valid, t, max_range)

    readings = np.min(t, axis=1)                       # (N,)


    return readings / max_range


def batch_get_inputs(track, xs, ys, angles, speeds, next_cps):
    """Vectorized input computation for all alive cars at once."""
    n = len(xs)

    # Filter segments to those near any alive car
    pad = float(MAX_SENSOR_RANGE)
    seg = track.seg_array
    keep = ((np.maximum(seg[:, 0], seg[:, 2]) >= xs.min() - pad) &
            (np.minimum(seg[:, 0], seg[:, 2]) <= xs.max() + pad) &
            (np.maximum(seg[:, 1], seg[:, 3]) >= ys.min() - pad) &
            (np.minimum(seg[:, 1], seg[:, 3]) <= ys.max() + pad))
    fseg = seg[keep]

    if len(fseg) == 0:
        sensors = np.ones((n, SENSOR_COUNT), dtype=np.float32)
    else:
        ray_angles = angles[:, None] + SENSOR_OFFSETS[None, :]
        cos_a = np.cos(ray_angles).ravel().astype(np.float32)
        sin_a = np.sin(ray_angles).ravel().astype(np.float32)
        ox = np.repeat(xs, SENSOR_COUNT)
        oy = np.repeat(ys, SENSOR_COUNT)

        fsx = fseg[:, 2] - fseg[:, 0]
        fsy = fseg[:, 3] - fseg[:, 1]

        dox = fseg[None, :, 0] - ox[:, None]
        doy = fseg[None, :, 1] - oy[:, None]

        denom = cos_a[:, None] * fsy[None, :] - sin_a[:, None] * fsx[None, :]
        t_num = dox * fsy[None, :] - doy * fsx[None, :]
        u_num = dox * sin_a[:, None] - doy * cos_a[:, None]

        abs_denom = np.abs(denom)
        safe = np.where(abs_denom > 1e-9, denom, 1.0)
        t = t_num / safe
        u = u_num / safe

        valid = (abs_denom > 1e-9) & (t > 0) & (t < MAX_SENSOR_RANGE) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, float(MAX_SENSOR_RANGE))
        sensors = (np.min(t, axis=1) / MAX_SENSOR_RANGE).reshape(n, SENSOR_COUNT)

    # Lookahead (vectorized across all cars)
    lookahead = np.empty((n, LOOKAHEAD_COUNT * 2), dtype=np.float32)
    for la in range(LOOKAHEAD_COUNT):
        cp_idx = (next_cps + la) % CHECKPOINT_COUNT
        cp_xy = track.cp_centers_np[cp_idx]
        a2cp = np.arctan2(cp_xy[:, 1] - ys, cp_xy[:, 0] - xs)
        rel = a2cp - angles
        rel = np.where(rel > np.pi, rel - 2 * np.pi, rel)
        rel = np.where(rel < -np.pi, rel + 2 * np.pi, rel)
        dist = np.hypot(cp_xy[:, 0] - xs, cp_xy[:, 1] - ys)
        lookahead[:, la * 2] = rel / np.pi
        lookahead[:, la * 2 + 1] = np.minimum(1.0, dist / 400.0)

    # Danger (vectorized across all cars)
    dangers = np.zeros(n, dtype=np.float32)
    for px, py, count in track.crash_points:
        d2 = (px - xs) ** 2 + (py - ys) ** 2
        mask = d2 < CRASH_MEMORY_RADIUS * CRASH_MEMORY_RADIUS
        if np.any(mask):
            danger = (1.0 - np.sqrt(d2[mask]) / CRASH_MEMORY_RADIUS) * min(1.0, count / 5.0)
            dangers[mask] = np.maximum(dangers[mask], danger)

    return np.column_stack([
        sensors, lookahead,
        speeds / MAX_SPEED,
        next_cps.astype(np.float32) / CHECKPOINT_COUNT,
        dangers
    ])



class Car:

    __slots__ = ['track', 'x', 'y', 'angle', 'speed', 'alive', 'fitness',

                 'brain', 'next_cp', 'laps', 'prev_x', 'prev_y', 'idle_time', 'idx']

   

    def __init__(self, track, brain=None, idx=0):

        self.track = track

        self.x = track.spawn_x

        self.y = track.spawn_y

        self.angle = track.spawn_angle

        self.speed = 0.0

        self.alive = True

        self.fitness = 0.0

        self.brain = brain if brain else NeuralNet()

        self.next_cp = 1

        self.laps = 0

        self.prev_x = self.x

        self.prev_y = self.y

        self.idle_time = 0.0

        self.idx = idx


    def get_inputs(self):

        # Fast sensor casting using spatial grid

        nearby = self.track.get_nearby_segments(self.x, self.y, MAX_SENSOR_RANGE)

        if nearby:

            nearby_segs = self.track.seg_array[list(nearby)]

        else:

            nearby_segs = np.empty((0, 4), dtype=np.float32)


        angles = self.angle + np.linspace(-1.0, 1.0, SENSOR_COUNT)

        sensors = batch_ray_intersect(self.x, self.y, angles, nearby_segs, MAX_SENSOR_RANGE)

       

        # Lookahead

        lookahead = np.empty(LOOKAHEAD_COUNT * 2, dtype=np.float32)

        for i in range(LOOKAHEAD_COUNT):

            cp_x, cp_y = self.track.cp_centers[(self.next_cp + i) % CHECKPOINT_COUNT]

            angle_to_cp = math.atan2(cp_y - self.y, cp_x - self.x)

            rel_angle = angle_to_cp - self.angle

            if rel_angle > math.pi: rel_angle -= 2 * math.pi

            elif rel_angle < -math.pi: rel_angle += 2 * math.pi

            dist = math.hypot(cp_x - self.x, cp_y - self.y)

            lookahead[i * 2] = rel_angle / math.pi

            lookahead[i * 2 + 1] = min(1.0, dist / 400.0)

       

        danger = self.track.get_danger_level(self.x, self.y)

       

        return np.concatenate([

            sensors, lookahead,

            np.array([self.speed / MAX_SPEED, (self.next_cp / CHECKPOINT_COUNT), danger], dtype=np.float32)

        ])


    def apply_outputs(self, outputs, dt):

        if not self.alive:

            return

       

        steer = float(outputs[0])

        throttle = (float(outputs[1]) + 1) * 0.5

        brake = max(0, float(outputs[2]))

       

        self.angle += steer * 3.5 * dt

        accel = throttle * MAX_SPEED - brake * MAX_SPEED * 1.5

        self.speed = max(0.5, min(MAX_SPEED, self.speed + accel * dt * 2.5))

       

        self.prev_x, self.prev_y = self.x, self.y

        self.x += math.cos(self.angle) * self.speed * dt * 60

        self.y += math.sin(self.angle) * self.speed * dt * 60

       

        # Fitness: progress toward next checkpoint + speed bonus
        cp_x, cp_y = self.track.cp_centers[self.next_cp]
        prev_dist = math.hypot(cp_x - self.prev_x, cp_y - self.prev_y)
        curr_dist = math.hypot(cp_x - self.x, cp_y - self.y)
        progress = prev_dist - curr_dist
        if progress > 0:
            self.fitness += progress * 2.0
        else:
            self.fitness += progress * 0.5

        speed_ratio = self.speed / MAX_SPEED
        self.fitness += speed_ratio * 0.5 * dt * 60

        if self.speed < MAX_SPEED * 0.3:
            self.fitness -= 0.3 * dt * 60

        # Alignment bonus: only rewards steering when actually moving
        angle_to_cp = math.atan2(cp_y - self.y, cp_x - self.x)
        alignment = math.cos(angle_to_cp - self.angle)
        self.fitness += alignment * speed_ratio * 0.3 * dt * 60

        # Idle check
        if self.speed < 2:

            self.idle_time += dt

            if self.idle_time > IDLE_TIMEOUT:

                self.alive = False

        else:

            self.idle_time = max(0, self.idle_time - dt * 2)

        # Checkpoint (track bounds checked in batch after all cars update)
        if math.hypot(self.x - cp_x, self.y - cp_y) < 30:

            self.next_cp = (self.next_cp + 1) % CHECKPOINT_COUNT

            self.fitness += 150.0

            if self.next_cp == 0:

                self.laps += 1

                self.fitness += 1000.0


    def draw(self, screen, camera, is_leader=False):

        sx, sy = camera.world_to_screen(self.x, self.y)

        if not (-20 < sx < WIDTH + 20 and -20 < sy < HEIGHT + 20):

            return


        if is_leader:

            color = (255, 70, 70)

        elif self.alive:

            color = (80, 210, 80)

        else:

            color = (150, 150, 150)


        sz = max(4, camera.zoom * 3)

        ca, sa = math.cos(self.angle), math.sin(self.angle)

        # Triangle pointing in direction of travel

        tip = (sx + ca * sz * 2, sy + sa * sz * 2)

        bl = (sx + (-ca - sa) * sz, sy + (-sa + ca) * sz)

        br = (sx + (-ca + sa) * sz, sy + (-sa - ca) * sz)

        pygame.draw.polygon(screen, color, [tip, bl, br])



def evolve(cars, track, stagnation_count=0):

    cars.sort(key=lambda c: c.fitness, reverse=True)

    best_fitness = cars[0].fitness

    best_brain = cars[0].brain

    second_best = cars[1].brain if len(cars) > 1 else best_brain


    survivor_count = max(5, POPULATION // 4)

    survivors = cars[:survivor_count]


    mutation_rate = BASE_MUTATION_RATE

    if stagnation_count >= STAGNATION_GENS:

        mutation_rate *= 2.5


    new_cars = []

    # Elites (exact copies)

    for i in range(min(3, len(survivors))):

        new_cars.append(Car(track, survivors[i].brain.copy(), len(new_cars)))


    # 60% mutations of best

    for _ in range(POPULATION * 60 // 100):

        if len(new_cars) >= POPULATION: break

        brain = best_brain.copy()

        brain.mutate(mutation_rate * 0.5)

        new_cars.append(Car(track, brain, len(new_cars)))


    # 20% mutations of second best

    for _ in range(POPULATION * 20 // 100):

        if len(new_cars) >= POPULATION: break

        brain = second_best.copy()

        brain.mutate(mutation_rate * 0.7)

        new_cars.append(Car(track, brain, len(new_cars)))


    # Rest from top survivors

    while len(new_cars) < POPULATION:

        weights = [1.0 / (i + 1) ** 2 for i in range(len(survivors))]

        parent = random.choices(survivors, weights=weights, k=1)[0]

        brain = parent.brain.copy()

        brain.mutate(mutation_rate)

        new_cars.append(Car(track, brain, len(new_cars)))


    return new_cars, best_fitness



def draw_hud(gen, alive, stagnation, leader, best_ever, gen_time, fps):

    """Semi-transparent HUD with generation info, leader stats, and progress bars."""

    hud = pygame.Surface((WIDTH, 54), pygame.SRCALPHA)

    hud.fill((10, 10, 15, 200))

    screen.blit(hud, (0, 0))


    # Line 1: generation info

    l1 = f"GEN {gen}   ALIVE {alive}/{POPULATION}   STAG {stagnation}/{STAGNATION_GENS}   FPS {fps:.0f}"

    screen.blit(font.render(l1, True, (200, 200, 200)), (10, 4))


    # Line 2: leader stats

    if leader.alive:

        spd_pct = leader.speed / MAX_SPEED * 100

        spd_col = ((100, 255, 100) if spd_pct > 50 else

                   (255, 200, 100) if spd_pct > 30 else

                   (255, 100, 100))

        l2 = (f"CP {leader.next_cp}/{CHECKPOINT_COUNT}  LAP {leader.laps}  "

              f"SPD {spd_pct:.0f}%  FIT {leader.fitness:.0f}  BEST {best_ever:.0f}")

    else:

        spd_col = (120, 120, 120)

        l2 = f"(no leader alive)  FIT {leader.fitness:.0f}  BEST {best_ever:.0f}"

    screen.blit(font.render(l2, True, spd_col), (10, 26))


    # Generation timer bar (top right)

    bar_w = 120

    bar_x = WIDTH - bar_w - 15

    progress = min(1.0, gen_time / GENERATION_TIME)

    pygame.draw.rect(screen, (40, 40, 45), (bar_x, 6, bar_w, 10))

    pygame.draw.rect(screen, (80, 200, 120), (bar_x, 6, int(bar_w * progress), 10))

    screen.blit(font_sm.render(f"{gen_time:.0f}s", True, (160, 160, 160)), (bar_x + bar_w + 4, 3))


    # Checkpoint progress bar

    if leader.alive:

        cp_prog = leader.next_cp / CHECKPOINT_COUNT

        pygame.draw.rect(screen, (40, 40, 45), (bar_x, 22, bar_w, 10))

        pygame.draw.rect(screen, (100, 180, 255), (bar_x, 22, int(bar_w * cp_prog), 10))

        screen.blit(font_sm.render(f"{cp_prog * 100:.0f}%", True, (160, 160, 160)),

                    (bar_x + bar_w + 4, 19))



def show_track_menu():

    tracks_dir = os.path.join(os.path.dirname(__file__), "tracks")

    if not os.path.exists(tracks_dir):

        print(f"Tracks folder not found: {tracks_dir}")

        pygame.quit()

        exit()

   

    available = sorted([f[:-4] for f in os.listdir(tracks_dir) if f.endswith(".csv")])

    if not available:

        print("No track CSV files found in tracks folder!")

        pygame.quit()

        exit()

   

    selected = 0

   

    while True:

        screen.fill((25, 30, 38))

        title = font_large.render("SELECT TRACK", True, (255, 220, 100))

        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

       

        hint = font.render("UP/DOWN to select, ENTER to start, ESC to quit", True, (120, 120, 120))

        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, 60))

       

        # Scrollable list if many tracks

        visible_start = max(0, selected - 10)

        visible_end = min(len(available), visible_start + 20)

       

        for i, idx in enumerate(range(visible_start, visible_end)):

            name = available[idx]

            color = (255, 255, 100) if idx == selected else (180, 180, 180)

            prefix = "> " if idx == selected else "  "

            txt = font.render(f"{prefix}{name}", True, color)

            screen.blit(txt, (50, 100 + i * 24))

       

        # Track count

        count_txt = font.render(f"{len(available)} tracks available", True, (100, 100, 100))

        screen.blit(count_txt, (WIDTH - 180, HEIGHT - 30))

       

        pygame.display.flip()

        clock.tick(30)

       

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()

                exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:

                    selected = (selected - 1) % len(available)

                elif event.key == pygame.K_DOWN:

                    selected = (selected + 1) % len(available)

                elif event.key == pygame.K_RETURN:

                    return available[selected]

                elif event.key == pygame.K_ESCAPE:

                    pygame.quit()

                    exit()



def main():

    track_name = show_track_menu()

    track = Track(track_name)

    pygame.display.set_caption(f"Neural Racing - {track_name}")


    camera = Camera()

    camera.x, camera.y = track.spawn_x, track.spawn_y


    cars = [Car(track, idx=i) for i in range(POPULATION)]

    batched = BatchedBrains([c.brain for c in cars])


    generation = 1

    gen_time = 0.0

    best_ever = 0.0

    prev_best = 0.0

    stagnation = 0

    follow = True

    paused = False


    # Pre-allocate arrays reused every frame

    all_inputs = np.zeros((POPULATION, TOTAL_INPUTS), dtype=np.float32)

    alive_mask = np.zeros(POPULATION, dtype=bool)


    while True:

        dt = min(clock.tick(TARGET_FPS) / 1000.0, 0.04)


        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()

                return

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:

                    paused = not paused

                elif event.key == pygame.K_r:

                    cars = [Car(track, idx=i) for i in range(POPULATION)]

                    batched = BatchedBrains([c.brain for c in cars])

                    generation, gen_time, stagnation, prev_best = 1, 0.0, 0, 0.0

                elif event.key == pygame.K_f:

                    follow = not follow

                elif event.key == pygame.K_EQUALS:

                    camera.zoom_in()

                elif event.key == pygame.K_MINUS:

                    camera.zoom_out()

                elif event.key == pygame.K_ESCAPE:

                    pygame.quit()

                    return

            elif event.type == pygame.MOUSEWHEEL:

                camera.zoom_in() if event.y > 0 else camera.zoom_out()


        if paused:

            hud_bg = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

            hud_bg.fill((0, 0, 0, 100))

            screen.blit(hud_bg, (0, 0))

            pause_txt = font_large.render("PAUSED", True, (255, 255, 100))

            screen.blit(pause_txt, (WIDTH // 2 - pause_txt.get_width() // 2, HEIGHT // 2 - 12))

            pygame.display.flip()

            continue


        # Camera controls

        keys = pygame.key.get_pressed()

        spd = 300 / camera.zoom * dt

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:

            camera.x -= spd; camera.target_x = camera.x; follow = False

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:

            camera.x += spd; camera.target_x = camera.x; follow = False

        if keys[pygame.K_w] or keys[pygame.K_UP]:

            camera.y -= spd; camera.target_y = camera.y; follow = False

        if keys[pygame.K_s] or keys[pygame.K_DOWN]:

            camera.y += spd; camera.target_y = camera.y; follow = False


        leader = max(cars, key=lambda c: c.fitness)

        best_ever = max(best_ever, leader.fitness)


        if follow and leader.alive:

            camera.follow(leader.x, leader.y)

        camera.update(dt)


        # Draw track (cached surface)

        track.draw(camera, leader.next_cp if leader.alive else None)


        # Batched inference

        alive_mask[:] = False

        alive_cars = []

        for c in cars:

            if c.alive:

                alive_mask[c.idx] = True

                alive_cars.append(c)

        alive_count = len(alive_cars)


        if alive_count > 0:

            # Vectorized input computation for all alive cars at once
            a_xs = np.array([c.x for c in alive_cars], dtype=np.float32)
            a_ys = np.array([c.y for c in alive_cars], dtype=np.float32)
            a_ang = np.array([c.angle for c in alive_cars], dtype=np.float32)
            a_spd = np.array([c.speed for c in alive_cars], dtype=np.float32)
            a_cp = np.array([c.next_cp for c in alive_cars], dtype=np.int32)

            inputs = batch_get_inputs(track, a_xs, a_ys, a_ang, a_spd, a_cp)
            for j, c in enumerate(alive_cars):
                all_inputs[c.idx] = inputs[j]

            all_outputs = batched.forward(all_inputs, alive_mask)

            for c in alive_cars:

                c.apply_outputs(all_outputs[c.idx], dt)

            # Batch track bounds check (replaces 100 individual np.argmin calls)
            still_alive = [c for c in alive_cars if c.alive]
            if still_alive:
                sa_x = np.array([c.x for c in still_alive], dtype=np.float32)
                sa_y = np.array([c.y for c in still_alive], dtype=np.float32)
                on_track = track.batch_is_on_track(sa_x, sa_y)
                for j, c in enumerate(still_alive):
                    if not on_track[j]:
                        c.alive = False
                        track.record_crash(c.x, c.y)

            alive_count = sum(1 for c in cars if c.alive)


        # Draw cars

        for car in cars:

            car.draw(screen, camera, car is leader)


        # HUD

        draw_hud(generation, alive_count, stagnation, leader, best_ever, gen_time, clock.get_fps())


        pygame.display.flip()


        gen_time += dt

        if gen_time > GENERATION_TIME or alive_count == 0:

            current_best = max(c.fitness for c in cars)

            if current_best <= prev_best * 1.01:

                stagnation += 1

            else:

                stagnation = 0

            prev_best = max(prev_best, current_best)


            cars, gen_best = evolve(cars, track, stagnation)

            batched = BatchedBrains([c.brain for c in cars])

            best_ever = max(best_ever, gen_best)

            generation += 1

            gen_time = 0.0

            print(f"Gen {generation} | Best: {best_ever:.0f}")



if __name__ == "__main__":

    main() 