"""
F1 Track Neural Network Racing Simulation
PyTorch GPU-accelerated version with batched neural network inference.
"""

import pygame
import numpy as np
import math
import random
import os
import torch

# Better CUDA detection with diagnostics
def setup_device():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA not available. Possible reasons:")
        print("  - PyTorch CPU-only version installed")
        print("  - Try: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return torch.device("cpu")

device = setup_device()
# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 1000, 600
POPULATION = 20  # Good balance for learning
SENSOR_COUNT = 8
LOOKAHEAD_COUNT = 3  # Number of checkpoints to look ahead
MAX_SENSOR_RANGE = 120
MAX_SPEED = 30
BASE_MUTATION_RATE = 0.1  # Reduced for more stable learning
GENERATION_TIME = 30.0  # seconds
CHECKPOINT_COUNT = 45
TARGET_FPS = 120
STAGNATION_GENS = 8  # More patience before exploring
# Inputs: sensors + lookahead(angle,dist)*3 + speed + lap_progress + danger_ahead
TOTAL_INPUTS = SENSOR_COUNT + LOOKAHEAD_COUNT * 2 + 3  # +1 for danger input
HIDDEN_SIZE = 32  # Bigger network since GPU handles it
CRASH_MEMORY_RADIUS = 80  # How close to a crash point to consider "danger zone"
# ---------------------------------------

import os
os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center window

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 Neural Racing")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Consolas", 18)
font_large = pygame.font.SysFont("Consolas", 26, bold=True)


class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 0.6
        self.min_zoom = 0.08
        self.max_zoom = 4.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_zoom = 0.6
    
    def follow(self, x, y):
        self.target_x = x
        self.target_y = y
    
    def update(self, dt):
        # Smooth interpolation
        t = min(1.0, dt * 8.0)
        self.x += (self.target_x - self.x) * t
        self.y += (self.target_y - self.y) * t
        self.zoom += (self.target_zoom - self.zoom) * t
    
    def zoom_in(self):
        self.target_zoom = min(self.max_zoom, self.target_zoom * 1.3)
    
    def zoom_out(self):
        self.target_zoom = max(self.min_zoom, self.target_zoom / 1.3)
    
    def world_to_screen(self, wx, wy):
        return (
            (wx - self.x) * self.zoom + WIDTH * 0.5,
            (wy - self.y) * self.zoom + HEIGHT * 0.5
        )


# Fast intersection using pure python (avoiding numpy overhead for small ops)
def _ray_seg_intersect(ox, oy, dx, dy, ax, ay, bx, by):
    """Return distance or None."""
    sx, sy = bx - ax, by - ay
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None
    t = ((ax - ox) * sy - (ay - oy) * sx) / denom
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def _seg_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
    """Fast segment intersection check."""
    d1x, d1y = p2x - p1x, p2y - p1y
    d2x, d2y = q2x - q1x, q2y - q1y
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-9:
        return False
    t = ((q1x - p1x) * d2y - (q1y - p1y) * d2x) / cross
    u = ((q1x - p1x) * d1y - (q1y - p1y) * d1x) / cross
    return 0 <= t <= 1 and 0 <= u <= 1


class Track:
    def __init__(self, track_name):
        self.name = track_name
        self._load_track_csv(track_name)
        self._compute_boundaries()
        self._build_segments()
        self._generate_checkpoints()
        self._compute_spawn()
        # Crash memory: track where cars frequently crash
        self.crash_points = []  # List of (x, y, count)
        self.crash_heatmap = {}  # Grid-based crash counting

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
                cy.append(-float(parts[1]))  # Flip Y
                wr.append(float(parts[2]) if len(parts) > 2 else 6.0)
                wl.append(float(parts[3]) if len(parts) > 3 else 6.0)
        
        # Center around origin
        cx = np.array(cx)
        cy = np.array(cy)
        cx -= (cx.min() + cx.max()) / 2
        cy -= (cy.min() + cy.max()) / 2
        
        self.center_x = cx
        self.center_y = cy
        # Scale track width slightly (1.5x original for more room)
        self.w_right = np.array(wr) * 5
        self.w_left = np.array(wl) * 5
        self.n = len(cx)

    def _compute_boundaries(self):
        n = self.n
        self.left_x = np.zeros(n)
        self.left_y = np.zeros(n)
        self.right_x = np.zeros(n)
        self.right_y = np.zeros(n)
        
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            dx = self.center_x[next_i] - self.center_x[prev_i]
            dy = self.center_y[next_i] - self.center_y[prev_i]
            length = math.hypot(dx, dy) + 1e-9
            nx, ny = -dy / length, dx / length
            
            self.left_x[i] = self.center_x[i] + nx * self.w_left[i]
            self.left_y[i] = self.center_y[i] + ny * self.w_left[i]
            self.right_x[i] = self.center_x[i] - nx * self.w_right[i]
            self.right_y[i] = self.center_y[i] - ny * self.w_right[i]

    def _build_segments(self):
        # Pre-build segment list as tuples for fast iteration
        n = self.n
        self.segments = []
        for i in range(n):
            ni = (i + 1) % n
            self.segments.append((self.left_x[i], self.left_y[i], self.left_x[ni], self.left_y[ni]))
            self.segments.append((self.right_x[i], self.right_y[i], self.right_x[ni], self.right_y[ni]))

    def _generate_checkpoints(self):
        self.checkpoints = []
        for k in range(CHECKPOINT_COUNT):
            idx = int((k / CHECKPOINT_COUNT) * self.n) % self.n
            self.checkpoints.append((
                self.left_x[idx], self.left_y[idx],
                self.right_x[idx], self.right_y[idx]
            ))

    def _compute_spawn(self):
        cp = self.checkpoints[0]
        self.spawn_x = (cp[0] + cp[2]) / 2
        self.spawn_y = (cp[1] + cp[3]) / 2
        
        idx = 0
        next_idx = int((1 / CHECKPOINT_COUNT) * self.n) % self.n
        dx = self.center_x[next_idx] - self.center_x[idx]
        dy = self.center_y[next_idx] - self.center_y[idx]
        self.spawn_angle = math.atan2(dy, dx)

    def is_on_track(self, x, y):
        # Find closest center point
        dx = self.center_x - x
        dy = self.center_y - y
        dists = dx * dx + dy * dy
        idx = np.argmin(dists)
        max_w = (self.w_left[idx] + self.w_right[idx]) / 2 + 2
        return dists[idx] < max_w * max_w

    def draw(self, camera, next_cp=None):
        # Draw track as individual quad segments to avoid self-intersection
        n = self.n
        for i in range(n):
            ni = (i + 1) % n
            quad = [
                camera.world_to_screen(self.left_x[i], self.left_y[i]),
                camera.world_to_screen(self.left_x[ni], self.left_y[ni]),
                camera.world_to_screen(self.right_x[ni], self.right_y[ni]),
                camera.world_to_screen(self.right_x[i], self.right_y[i]),
            ]
            pygame.draw.polygon(screen, (55, 55, 60), quad)
        
        # Draw boundaries as clean lines
        left_pts = [camera.world_to_screen(self.left_x[i], self.left_y[i]) for i in range(n)]
        right_pts = [camera.world_to_screen(self.right_x[i], self.right_y[i]) for i in range(n)]
        pygame.draw.lines(screen, (255, 255, 255), True, left_pts, 2)
        pygame.draw.lines(screen, (255, 255, 255), True, right_pts, 2)
        
        # Checkpoints
        for i, cp in enumerate(self.checkpoints):
            a = camera.world_to_screen(cp[0], cp[1])
            b = camera.world_to_screen(cp[2], cp[3])
            if next_cp is not None and i == next_cp:
                pygame.draw.line(screen, (0, 200, 255), a, b, 3)
            elif i == 0:
                pygame.draw.line(screen, (255, 60, 60), a, b, 3)
            else:
                pygame.draw.line(screen, (70, 70, 70), a, b, 1)

    def record_crash(self, x, y):
        """Record a crash location to build danger zone memory."""
        # Grid-based tracking (50 unit cells)
        grid_x = int(x / 50)
        grid_y = int(y / 50)
        key = (grid_x, grid_y)
        self.crash_heatmap[key] = self.crash_heatmap.get(key, 0) + 1
        
        # Also store exact point if it's a hotspot
        if self.crash_heatmap[key] >= 3:  # 3+ crashes in same area
            # Check if already have a point nearby
            for i, (px, py, count) in enumerate(self.crash_points):
                if math.hypot(px - x, py - y) < CRASH_MEMORY_RADIUS:
                    # Update existing point
                    self.crash_points[i] = (px, py, count + 1)
                    return
            # New crash hotspot
            self.crash_points.append((x, y, 1))
    
    def get_danger_level(self, x, y):
        """Get danger level (0-1) based on proximity to known crash points."""
        if not self.crash_points:
            return 0.0
        
        max_danger = 0.0
        for px, py, count in self.crash_points:
            dist = math.hypot(px - x, py - y)
            if dist < CRASH_MEMORY_RADIUS:
                # Closer = more danger, more crashes = more danger
                danger = (1.0 - dist / CRASH_MEMORY_RADIUS) * min(1.0, count / 5.0)
                max_danger = max(max_danger, danger)
        return max_danger
    
    def draw_danger_zones(self, camera):
        """Draw crash hotspots on the track."""
        for px, py, count in self.crash_points:
            sx, sy = camera.world_to_screen(px, py)
            radius = max(5, min(20, count * 3)) * camera.zoom
            alpha = min(150, count * 30)
            # Draw red circle for danger zone
            surf = pygame.Surface((int(radius * 2), int(radius * 2)), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 0, 0, alpha), (int(radius), int(radius)), int(radius))
            screen.blit(surf, (sx - radius, sy - radius))


class NeuralNet:
    """PyTorch-based neural network running on GPU."""
    def __init__(self):
        self.w1 = torch.randn(TOTAL_INPUTS, HIDDEN_SIZE, device=device) * 0.5
        self.b1 = torch.zeros(HIDDEN_SIZE, device=device)
        self.w2 = torch.randn(HIDDEN_SIZE, 3, device=device) * 0.5  # steer, throttle, brake
        self.b2 = torch.zeros(3, device=device)

    def forward(self, x):
        """x can be a single input or batched inputs."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        h = torch.tanh(x @ self.w1 + self.b1)
        return torch.tanh(h @ self.w2 + self.b2)

    def mutate(self, rate=BASE_MUTATION_RATE):
        with torch.no_grad():
            for w in [self.w1, self.w2]:
                mask = torch.rand_like(w) < 0.3
                w[mask] += torch.randn(mask.sum().item(), device=device) * rate
            for b in [self.b1, self.b2]:
                mask = torch.rand_like(b) < 0.3
                b[mask] += torch.randn(mask.sum().item(), device=device) * rate

    def copy(self):
        new = NeuralNet()
        new.w1 = self.w1.clone()
        new.w2 = self.w2.clone()
        new.b1 = self.b1.clone()
        new.b2 = self.b2.clone()
        return new


class BatchedBrains:
    """All car brains batched together for parallel GPU inference."""
    def __init__(self, brains):
        self.n = len(brains)
        # Stack all weights into batched tensors [n, input, hidden] etc
        self.w1 = torch.stack([b.w1 for b in brains])  # [n, TOTAL_INPUTS, HIDDEN]
        self.b1 = torch.stack([b.b1 for b in brains])  # [n, HIDDEN]
        self.w2 = torch.stack([b.w2 for b in brains])  # [n, HIDDEN, 3]
        self.b2 = torch.stack([b.b2 for b in brains])  # [n, 3]
    
    @torch.no_grad()
    def forward_all(self, inputs):
        """
        inputs: numpy array [n, TOTAL_INPUTS]
        returns: numpy array [n, 3] (steer, throttle, brake for each car)
        """
        x = torch.tensor(inputs, dtype=torch.float32, device=device)  # [n, TOTAL_INPUTS]
        # Batched matrix multiply: [n, 1, TOTAL_INPUTS] @ [n, TOTAL_INPUTS, HIDDEN] -> [n, 1, HIDDEN]
        h = torch.tanh(torch.bmm(x.unsqueeze(1), self.w1).squeeze(1) + self.b1)
        out = torch.tanh(torch.bmm(h.unsqueeze(1), self.w2).squeeze(1) + self.b2)
        return out.cpu().numpy()


class Car:
    def __init__(self, track, brain=None):
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

    def cast_sensors(self):
        readings = np.empty(SENSOR_COUNT, dtype=np.float32)
        angles = np.linspace(-1.2, 1.2, SENSOR_COUNT)
        
        for i, a in enumerate(angles):
            dx = math.cos(self.angle + a)
            dy = math.sin(self.angle + a)
            best = MAX_SENSOR_RANGE
            
            for seg in self.track.segments:
                t = _ray_seg_intersect(self.x, self.y, dx, dy, seg[0], seg[1], seg[2], seg[3])
                if t is not None and 0 < t < best:
                    best = t
            
            readings[i] = best / MAX_SENSOR_RANGE
        return readings

    def get_lookahead(self):
        """Get angle and distance to upcoming checkpoints."""
        lookahead = np.empty(LOOKAHEAD_COUNT * 2, dtype=np.float32)
        
        for i in range(LOOKAHEAD_COUNT):
            cp_idx = (self.next_cp + i) % CHECKPOINT_COUNT
            cp = self.track.checkpoints[cp_idx]
            # Checkpoint center
            cp_x = (cp[0] + cp[2]) / 2
            cp_y = (cp[1] + cp[3]) / 2
            
            # Relative angle to checkpoint (normalized to -1 to 1)
            angle_to_cp = math.atan2(cp_y - self.y, cp_x - self.x)
            rel_angle = angle_to_cp - self.angle
            # Normalize to [-pi, pi]
            while rel_angle > math.pi:
                rel_angle -= 2 * math.pi
            while rel_angle < -math.pi:
                rel_angle += 2 * math.pi
            
            # Distance (normalized)
            dist = math.hypot(cp_x - self.x, cp_y - self.y)
            max_dist = 500.0  # Reasonable max distance
            
            lookahead[i * 2] = rel_angle / math.pi  # -1 to 1
            lookahead[i * 2 + 1] = min(1.0, dist / max_dist)  # 0 to 1
        
        return lookahead

    def get_inputs(self):
        """Get all inputs for this car (for batched inference)."""
        sensors = self.cast_sensors()
        lookahead = self.get_lookahead()
        speed_input = self.speed / MAX_SPEED
        lap_progress = (self.laps + self.next_cp / CHECKPOINT_COUNT)
        progress_input = (lap_progress % 1.0)
        
        # Danger level from crash memory
        danger_level = self.track.get_danger_level(self.x, self.y)
        
        return np.concatenate([
            sensors, 
            lookahead, 
            np.array([speed_input, progress_input, danger_level], dtype=np.float32)
        ])

    def apply_outputs(self, outputs, dt):
        """Apply neural network outputs to update car physics."""
        if not self.alive:
            return
            
        steer = float(outputs[0])
        throttle = (float(outputs[1]) + 1) / 2  # 0 to 1
        brake = max(0, float(outputs[2]))  # 0 to 1 (only positive braking)
        
        # Physics with delta time
        self.angle += steer * 4.0 * dt
        
        # Throttle accelerates, brake decelerates
        accel = throttle * MAX_SPEED - brake * MAX_SPEED * 1.5
        target_speed = self.speed + accel * dt * 3
        target_speed = max(0.5, min(MAX_SPEED, target_speed))
        self.speed += (target_speed - self.speed) * 6.0 * dt
        self.speed = max(0.5, min(MAX_SPEED, self.speed))
        
        self.prev_x, self.prev_y = self.x, self.y
        self.x += math.cos(self.angle) * self.speed * dt * 60
        self.y += math.sin(self.angle) * self.speed * dt * 60
        
        # Reward moving TOWARD the next checkpoint (goal-directed fitness)
        cp = self.track.checkpoints[self.next_cp]
        cp_x = (cp[0] + cp[2]) / 2
        cp_y = (cp[1] + cp[3]) / 2
        
        prev_dist_to_cp = math.hypot(cp_x - self.prev_x, cp_y - self.prev_y)
        curr_dist_to_cp = math.hypot(cp_x - self.x, cp_y - self.y)
        progress_toward_goal = prev_dist_to_cp - curr_dist_to_cp
        
        if progress_toward_goal > 0:
            self.fitness += progress_toward_goal * 2.0
        else:
            self.fitness += progress_toward_goal * 0.5
        
        # Idle check
        if self.speed < 2:
            self.idle_time += dt
            if self.idle_time > 1.5:
                self.alive = False
        else:
            self.idle_time = 0
        
        # Track bounds
        if not self.track.is_on_track(self.x, self.y):
            self.alive = False
            return
        
        # Checkpoint
        cp = self.track.checkpoints[self.next_cp]
        if _seg_intersect(self.prev_x, self.prev_y, self.x, self.y, cp[0], cp[1], cp[2], cp[3]):
            self.next_cp += 1
            self.fitness += 200.0
            if self.next_cp >= CHECKPOINT_COUNT:
                self.next_cp = 0
                self.laps += 1
                self.fitness += 1000.0

    def update(self, dt):
        """Legacy single-car update (not used in batched mode)."""
        if not self.alive:
            return
        
        # Combine wall sensors with checkpoint lookahead, speed, progress, and danger
        sensors = self.cast_sensors()
        lookahead = self.get_lookahead()
        speed_input = self.speed / MAX_SPEED  # Normalized speed
        lap_progress = (self.laps + self.next_cp / CHECKPOINT_COUNT)
        progress_input = (lap_progress % 1.0)  # Current lap progress 0-1
        danger_level = self.track.get_danger_level(self.x, self.y)
        
        inputs = np.concatenate([
            sensors, 
            lookahead, 
            np.array([speed_input, progress_input, danger_level], dtype=np.float32)
        ])
        
        outputs = self.brain.forward(inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        steer = outputs[0]
        throttle = (outputs[1] + 1) / 2  # 0 to 1
        brake = max(0, outputs[2])  # 0 to 1 (only positive braking)
        
        # Physics with delta time
        self.angle += steer * 4.0 * dt
        
        # Throttle accelerates, brake decelerates
        accel = throttle * MAX_SPEED - brake * MAX_SPEED * 1.5
        target_speed = self.speed + accel * dt * 3
        target_speed = max(0.5, min(MAX_SPEED, target_speed))
        self.speed += (target_speed - self.speed) * 6.0 * dt
        self.speed = max(0.5, min(MAX_SPEED, self.speed))
        
        self.prev_x, self.prev_y = self.x, self.y
        self.x += math.cos(self.angle) * self.speed * dt * 60
        self.y += math.sin(self.angle) * self.speed * dt * 60
        
        # Reward moving TOWARD the next checkpoint (goal-directed fitness)
        cp = self.track.checkpoints[self.next_cp]
        cp_x = (cp[0] + cp[2]) / 2
        cp_y = (cp[1] + cp[3]) / 2
        
        prev_dist_to_cp = math.hypot(cp_x - self.prev_x, cp_y - self.prev_y)
        curr_dist_to_cp = math.hypot(cp_x - self.x, cp_y - self.y)
        progress_toward_goal = prev_dist_to_cp - curr_dist_to_cp  # Positive if getting closer
        
        # Fitness: reward progress toward checkpoint (not just any movement)
        if progress_toward_goal > 0:
            self.fitness += progress_toward_goal * 2.0  # Strong reward for goal progress
        else:
            self.fitness += progress_toward_goal * 0.5  # Mild penalty for moving away
        
        # Idle check
        if self.speed < 2:
            self.idle_time += dt
            if self.idle_time > 1.5:
                self.alive = False
        else:
            self.idle_time = 0
        
        # Track bounds - record crash in legacy update too
        if not self.track.is_on_track(self.x, self.y):
            self.alive = False
            self.track.record_crash(self.x, self.y)
            return
        
        # Checkpoint
        cp = self.track.checkpoints[self.next_cp]
        if _seg_intersect(self.prev_x, self.prev_y, self.x, self.y, cp[0], cp[1], cp[2], cp[3]):
            self.next_cp += 1
            self.fitness += 200.0
            if self.next_cp >= CHECKPOINT_COUNT:
                self.next_cp = 0
                self.laps += 1
                self.fitness += 1000.0

    def draw(self, camera, is_leader=False):
        sx, sy = camera.world_to_screen(self.x, self.y)
        
        # Skip if off screen
        if sx < -50 or sx > WIDTH + 50 or sy < -50 or sy > HEIGHT + 50:
            return
        
        if self.alive:
            color = (255, 60, 60) if is_leader else (60, 220, 60)
        else:
            color = (80, 80, 80)
        
        # Small car shape (reduced from 5 to 1.5 base size)
        size = max(2, camera.zoom * 1.5)
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        pts = [
            (sx + cos_a * size * 2 - sin_a * size, sy + sin_a * size * 2 + cos_a * size),
            (sx + cos_a * size * 2 + sin_a * size, sy + sin_a * size * 2 - cos_a * size),
            (sx - cos_a * size + sin_a * size, sy - sin_a * size - cos_a * size),
            (sx - cos_a * size - sin_a * size, sy - sin_a * size + cos_a * size),
        ]
        pygame.draw.polygon(screen, color, pts)
        
        # Leader sensors
        if is_leader and self.alive:
            angles = np.linspace(-1.2, 1.2, SENSOR_COUNT)
            for a in angles:
                dx = math.cos(self.angle + a)
                dy = math.sin(self.angle + a)
                best = MAX_SENSOR_RANGE
                for seg in self.track.segments:
                    t = _ray_seg_intersect(self.x, self.y, dx, dy, seg[0], seg[1], seg[2], seg[3])
                    if t is not None and 0 < t < best:
                        best = t
                ex, ey = self.x + dx * best, self.y + dy * best
                pygame.draw.line(screen, (255, 220, 0), (sx, sy), camera.world_to_screen(ex, ey), 1)


def evolve(cars, track, stagnation_count=0):
    cars.sort(key=lambda c: c.fitness, reverse=True)
    best_fitness = cars[0].fitness
    best_brain = cars[0].brain
    
    # Keep top 50% as survivors (more elitism)
    survivor_count = max(2, POPULATION // 2)
    survivors = cars[:survivor_count]
    
    # Adaptive mutation based on stagnation - but MUCH less aggressive
    if stagnation_count >= STAGNATION_GENS:
        mutation_rate = BASE_MUTATION_RATE * 2  # Only double, not triple
        explore_count = 2  # Just 2 explorers, not 25%
    else:
        mutation_rate = BASE_MUTATION_RATE
        explore_count = 0  # No random exploration normally
    
    new_cars = []
    
    # Keep top 3 unchanged (strong elitism)
    for i in range(min(3, len(survivors))):
        new_cars.append(Car(track, survivors[i].brain.copy()))
    
    # Breed rest from survivors with mild mutation
    while len(new_cars) < POPULATION - explore_count:
        # Prefer better parents (weighted selection)
        weights = [1.0 / (i + 1) for i in range(len(survivors))]
        parent = random.choices(survivors, weights=weights, k=1)[0]
        brain = parent.brain.copy()
        brain.mutate(mutation_rate)
        new_cars.append(Car(track, brain))
    
    # Explorers: mildly mutated copies of best (not heavily mutated!)
    for _ in range(explore_count):
        brain = best_brain.copy()
        brain.mutate(mutation_rate * 1.5)  # Only 1.5x, not 6x
        new_cars.append(Car(track, brain))
    
    return new_cars, best_fitness


def show_track_menu():
    tracks_dir = os.path.join(os.path.dirname(__file__), "tracks")
    available = sorted([f[:-4] for f in os.listdir(tracks_dir) if f.endswith(".csv")])
    selected = 0
    
    while True:
        screen.fill((25, 30, 38))
        
        title = font_large.render("F1 Neural Racing - Select Track", True, (255, 255, 255))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 25))
        
        cols = 4
        col_w = WIDTH // cols
        per_col = (len(available) + cols - 1) // cols
        
        for i, name in enumerate(available):
            col, row = i // per_col, i % per_col
            x, y = col * col_w + 40, 80 + row * 28
            
            if i == selected:
                pygame.draw.rect(screen, (50, 90, 140), (x - 6, y - 2, col_w - 60, 24), border_radius=3)
                color = (255, 255, 120)
            else:
                color = (160, 160, 160)
            
            screen.blit(font.render(name, True, color), (x, y))
        
        help_txt = font.render("UP/DOWN: Navigate | ENTER: Select | ESC: Quit", True, (90, 90, 90))
        screen.blit(help_txt, (WIDTH // 2 - help_txt.get_width() // 2, HEIGHT - 30))
        
        pygame.display.flip()
        
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


def draw_minimap(track, cars, leader):
    size = 140
    margin = 12
    
    # Bounds
    min_x, max_x = track.center_x.min(), track.center_x.max()
    min_y, max_y = track.center_y.min(), track.center_y.max()
    scale = (size - 16) / max(max_x - min_x, max_y - min_y)
    
    def to_map(x, y):
        return (x - min_x) * scale + margin + 8, (y - min_y) * scale + margin + 8
    
    pygame.draw.rect(screen, (30, 30, 38, 200), (margin, margin, size, size), border_radius=6)
    
    # Track outline
    pts = [to_map(track.center_x[i], track.center_y[i]) for i in range(track.n)]
    pygame.draw.lines(screen, (90, 90, 100), True, pts, 2)
    
    # Cars
    for car in cars:
        if car.alive:
            pos = to_map(car.x, car.y)
            col = (255, 80, 80) if car is leader else (80, 200, 80)
            pygame.draw.circle(screen, col, (int(pos[0]), int(pos[1])), 3)


def main():
    global WIDTH, HEIGHT, screen
    
    track_name = show_track_menu()
    track = Track(track_name)
    pygame.display.set_caption(f"F1 Neural Racing - {track_name}")
    
    camera = Camera()
    camera.x, camera.y = track.spawn_x, track.spawn_y
    
    cars = [Car(track) for _ in range(POPULATION)]
    generation = 1
    gen_time = 0.0
    best_ever = 0.0
    prev_best = 0.0
    stagnation = 0
    
    running = True
    paused = False
    follow = True
    
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        dt = min(dt, 0.05)  # Cap delta time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    cars = [Car(track) for _ in range(POPULATION)]
                    generation = 1
                    gen_time = 0.0
                    stagnation = 0
                    prev_best = 0.0
                elif event.key == pygame.K_f:
                    follow = not follow
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    camera.zoom_in()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    camera.zoom_out()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.zoom_in()
                else:
                    camera.zoom_out()
        
        if paused:
            txt = font_large.render("PAUSED - SPACE to continue", True, (255, 255, 0))
            screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2))
            pygame.display.flip()
            continue
        
        # Camera movement with WASD (works always, auto-disables follow)
        keys = pygame.key.get_pressed()
        moving = False
        spd = 400 / camera.zoom * dt
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            camera.x -= spd
            camera.target_x = camera.x
            moving = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            camera.x += spd
            camera.target_x = camera.x
            moving = True
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            camera.y -= spd
            camera.target_y = camera.y
            moving = True
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            camera.y += spd
            camera.target_y = camera.y
            moving = True
        
        if moving:
            follow = False  # Auto-disable follow when manually moving
        
        leader = max(cars, key=lambda c: c.fitness)
        best_ever = max(best_ever, leader.fitness)
        
        if follow and leader.alive:
            camera.follow(leader.x, leader.y)
        camera.update(dt)
        
        # Clear
        screen.fill((40, 75, 50))
        
        # Draw track
        track.draw(camera, leader.next_cp if leader.alive else None)
        
        # Draw danger zones (crash memory)
        track.draw_danger_zones(camera)
        
        # BATCHED GPU INFERENCE: Collect all alive car inputs, run through GPU in one batch
        alive_cars = [car for car in cars if car.alive]
        alive = len(alive_cars)
        
        if alive > 0:
            # Collect inputs from all alive cars
            all_inputs = np.array([car.get_inputs() for car in alive_cars], dtype=np.float32)
            
            # Create batched brains and run forward pass on GPU
            batched = BatchedBrains([car.brain for car in alive_cars])
            all_outputs = batched.forward_all(all_inputs)
            
            # Apply outputs to each car
            for car, outputs in zip(alive_cars, all_outputs):
                car.apply_outputs(outputs, dt)
        
        # Draw all cars
        for car in cars:
            car.draw(camera, car is leader)
        
        # Minimap
        draw_minimap(track, cars, leader)
        
        # HUD - top right
        hud_x = WIDTH - 400
        pygame.draw.rect(screen, (20, 20, 25, 220), (hud_x - 10, 8, 400, 115), border_radius=6)
        
        lines = [
            f"Track: {track.name}  |  Device: {device}",
            f"Gen: {generation}  |  Alive: {alive}/{POPULATION}",
            f"Fitness: {leader.fitness:.0f}  (Best: {best_ever:.0f})",
            f"Laps: {leader.laps}  |  Speed: {leader.speed:.1f}",
            f"Stagnation: {stagnation}{'  [EXPLORING]' if stagnation >= STAGNATION_GENS else ''}",
            f"Zoom: {camera.zoom:.2f}x  |  Follow: {'ON' if follow else 'OFF'}  |  FPS: {clock.get_fps():.0f}",
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (255, 255, 255)), (hud_x, 14 + i * 18))
        
        # Help
        help_txt = font.render("SPACE:Pause  R:Reset  F:Follow  +/-/Scroll:Zoom  WASD:Move  ESC:Quit", True, (130, 130, 130))
        screen.blit(help_txt, (WIDTH // 2 - help_txt.get_width() // 2, HEIGHT - 24))
        
        pygame.display.flip()
        
        gen_time += dt
        if gen_time > GENERATION_TIME or alive == 0:
            # Check for stagnation
            current_best = max(c.fitness for c in cars)
            if current_best <= prev_best * 1.01:  # Less than 1% improvement
                stagnation += 1
            else:
                stagnation = 0
            prev_best = max(prev_best, current_best)
            
            cars, gen_best = evolve(cars, track, stagnation)
            best_ever = max(best_ever, gen_best)
            generation += 1
            gen_time = 0.0
            
            status = "STAGNANT - Exploring!" if stagnation >= STAGNATION_GENS else ""
            print(f"Gen {generation} | Best: {best_ever:.0f} | Stagnation: {stagnation} {status}")
    
    pygame.quit()


if __name__ == "__main__":
    main()
