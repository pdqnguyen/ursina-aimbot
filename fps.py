from ursina import *
from ursina.shaders import lit_with_shadows_shader
from ursina import curve

from my_fpc import MyFPC


# Window parameters
FULL_SCREEN = False
FOV = 100
SCREENSHOT_MODE = False

# Stage parameters
STAGE_X = 64
STAGE_Y = 32
STAGE_Z = 64
WALL_THICKNESS = 5
WALL_COLOR = color.gray
WALL_TEXTURE = "forest"

# Game objects
NUM_OBSTACLES = 0
NUM_ENEMIES = 4
ENEMY_COLOR = rgb(74, 69, 24)
ENEMY_MAX_HP = 100

# Mouse parameters
CROSSHAIR = "crosshair"
MOUSE_MULTIPLIER = 1.35
MOUSE_SENSITIVITY = MOUSE_MULTIPLIER * Vec2(40, 40)

# Movement parameters
MOVEMENT_SPEED = 10
ACCELERATION = 3

# Gun parameters
FIRE_RATE = 800
RECOIL_VERT = 0.75
RECOIL_HORI = 0.5
BASE_DAMAGE = 10
HEADSHOT_MULTIPLIER = 2
ADS_ZOOM = 2
ADS_TIME = 0.1


app = Ursina()
window.position = (0, 0)
if FULL_SCREEN:
    window.size = window.fullscreen_size
else:
    window.size = window.fullscreen_size / 2.0

random.seed(0)
Entity.default_shader = lit_with_shadows_shader

# Stage elements
ground = Entity(model='plane', collider='box', scale_x=STAGE_X, scale_z=STAGE_Z, texture='grass', texture_scale=(4, 4))
wall_positions = [
    (0, 5, (STAGE_Z + WALL_THICKNESS) / 2),
    (0, 5, -(STAGE_Z + WALL_THICKNESS) / 2),
    ((STAGE_X + WALL_THICKNESS) / 2, 5, 0),
    (-(STAGE_X + WALL_THICKNESS) / 2, 5, 0),
]
wall_scales = [
    Vec3(STAGE_X, STAGE_Y, WALL_THICKNESS),
    Vec3(STAGE_X, STAGE_Y, WALL_THICKNESS),
    Vec3(WALL_THICKNESS, STAGE_Y, STAGE_Z),
    Vec3(WALL_THICKNESS, STAGE_Y, STAGE_Z),
]
walls = []
for position, scale in zip(wall_positions, wall_scales):
    wall = Entity(
        model='cube',
        color=WALL_COLOR,
        position=position,
        scale=scale,
        collider='box',
        texture=WALL_TEXTURE,
    )
    walls.append(wall)
obstacles = []
for i in range(NUM_OBSTACLES):
    obstacle = Entity(
        model='cube',
        origin_y=-0.5,
        scale=3,
        texture='brick',
        texture_scale=(1, 2),
        x=random.uniform(-STAGE_X / 2, STAGE_X / 2),
        z=random.uniform(-STAGE_Z / 2, STAGE_Z / 2),
        collider='box',
        scale_y=random.uniform(2, 3),
        color=color.hsv(0, 0, random.uniform(.9, 1))
    )
    obstacles.append(obstacle)


# Player and camera
editor_camera = EditorCamera(enabled=False, ignore_paused=True)
cursor = Entity(parent=camera.ui, model='quad', color=color.white, texture=CROSSHAIR, scale=.1)
if SCREENSHOT_MODE:
    cursor.alpha = 0
player = MyFPC(
    cursor,
    color=color.orange,
    alpha=0,
    speed=MOVEMENT_SPEED,
    mouse_sensitivity=MOUSE_SENSITIVITY,
    ads=False,
)
player.collider = BoxCollider(player, Vec3(0, 0, 0), Vec3(2, 2, 2))
camera.fov = FOV


gun = Entity(
    model='cube',
    parent=camera,
    position=(0.5, -0.25, 0.25),
    scale=(0.2, 0.2, 1),
    origin_z=-0.5,
    color=color.dark_gray,
    on_cooldown=False,
)
gun.muzzle_flash = Entity(parent=gun, z=1, world_scale=.4, model='quad', color=color.yellow, alpha=0.5, enabled=False)
if SCREENSHOT_MODE:
    gun.alpha = 0

shootables_parent = Entity()
mouse.traverse_target = shootables_parent


def update():
    if held_keys['left mouse']:
        shoot()
    elif SCREENSHOT_MODE:
        if random.uniform(0, 1) < 0.01:
            for enemy in enemies:
                if enemy.hp > 0:
                    enemy.hp = 0
                    new_enemy()
                    break
        if random.uniform(0, 1) < 0.1:
            player.rotation_y = random.uniform(0, 360)
        if random.uniform(0, 1) < 0.003:
            player.x = random.uniform(-STAGE_X / 2 + WALL_THICKNESS, STAGE_X / 2 - WALL_THICKNESS)
            player.z = random.uniform(-STAGE_Z / 2 + WALL_THICKNESS, STAGE_Z / 2 - WALL_THICKNESS)
        if random.uniform(0, 1) < 0.003:
            player.camera_pivot.rotation_x = random.uniform(-10, 10)


def ads(
        start,
        movement_speed=MOVEMENT_SPEED,
        mouse_sensitivity=MOUSE_SENSITIVITY,
        ads_zoom=ADS_ZOOM,
        ads_time=ADS_TIME
):
    if start:
        camera.animate('fov', FOV / ads_zoom, duration=ads_time, delay=0, curve=curve.linear)
        gun.animate_position((0, -0.25, 0.4), duration=ads_time)
        ads_sens = Vec2(mouse_sensitivity[0] / ads_zoom, mouse_sensitivity[1] / ads_zoom)
        invoke(setattr, player, 'mouse_sensitivity', ads_sens, delay=ads_time)
        invoke(setattr, player, 'speed', 0.5 * movement_speed, delay=ads_time)
        invoke(setattr, player, 'ads', True, delay=ads_time)
    else:
        camera.animate('fov', FOV, duration=ads_time, delay=0, curve=curve.linear)
        gun.animate_position((0.5, -0.25, 0.25), duration=ads_time)
        invoke(setattr, player, 'mouse_sensitivity', mouse_sensitivity, delay=ads_time)
        invoke(setattr, player, 'speed', movement_speed, delay=ads_time)
        invoke(setattr, player, 'ads', False, delay=ads_time)


def shoot(
        fire_rate=FIRE_RATE,
        recoil_vert=RECOIL_VERT,
        recoil_hori=RECOIL_HORI,
        base_damage=BASE_DAMAGE,
        headshot_multiplier=HEADSHOT_MULTIPLIER
):
    if not gun.on_cooldown:
        cooldown = 60. / fire_rate
        gun.on_cooldown = True
        gun.muzzle_flash.enabled = True
        if player.ads:
            visual_recoil = 2
        else:
            visual_recoil = 5
        gun.animate_rotation((-visual_recoil, 0, 0), 0.5 * cooldown, delay=0, curve=curve.linear)
        gun.animate_rotation((visual_recoil, 0, 0), 0.5 * cooldown, delay=0.5 * cooldown, curve=curve.linear)
        # from ursina.prefabs.ursfx import ursfx
        # ursfx(
        #     [(0.0, 0.0), (0.1, 0.9), (0.15, 0.75), (0.3, 0.14), (0.6, 0.0)],
        #     volume=0.3,
        #     wave='noise',
        #     pitch=random.uniform(-13, -12),
        #     pitch_change=-12,
        #     speed=3.0
        # )
        invoke(gun.muzzle_flash.disable, delay=cooldown)
        invoke(setattr, gun, 'on_cooldown', False, delay=cooldown)
        target = mouse.hovered_entity
        if target:
            damage = 0
            if hasattr(target, 'hp'):
                damage = base_damage
            elif hasattr(target.parent, 'hp'):
                target = target.parent
                damage = base_damage * headshot_multiplier
            if damage > 0:
                target.hp -= damage
                if target.hp <= 0:
                    new_enemy()
                target.blink(color.red)
        # Add recoil
        recoil_vert = random.uniform(0.8, 1.2) * recoil_vert * (camera.fov / FOV)
        recoil_hori = random.uniform(-1, 1) * recoil_hori * (camera.fov / FOV)
        player.camera_pivot.rotation_x -= recoil_vert
        player.rotation_y += recoil_hori
        # Alternative method, animated recoil... it sucks because it overrules mouse movements, making tracking hard
        # rot_x = player.camera_pivot.rotation_x - recoil_vert
        # rot_y = player.rotation_y + recoil_hori
        # player.camera_pivot.animate_rotation((rot_x, 0, 0), duration=0.01, curve=curve.linear)
        # player.animate_rotation((0, rot_y, 0), duration=0., curve=curve.linear)


class Enemy(Entity):
    def __init__(self, max_hp=ENEMY_MAX_HP, **kwargs):
        super().__init__(
            parent=shootables_parent,
            model="soldier",
            texture="soldier",
            scale=(1.5, 1.5, 1.5),
            **kwargs
        )
        self.head = Entity(
            parent=self,
            model='sphere',
            alpha=0,
            collider='sphere',
            position=(-0.08, 1.35, 0.1),
            scale=(0.2, 0.25, 0.2),
        )
        self.health_bar = Entity(parent=self, model='cube', color=color.red, y=1.6, world_scale=(1.5, 0.1, 0.1))
        self.max_hp = max_hp
        self.hp = self.max_hp
        self.speed = 0
        self.speed_y = 0
        self.rotation_y = random.uniform(0, 360)
        self.stationary = True

    def update(self, movement_speed=MOVEMENT_SPEED, acceleration=ACCELERATION):
        self.health_bar.alpha = max(0, self.health_bar.alpha - time.dt)
        self.health_bar.look_at_2d(player.position, axis='y')

        # For testing purposes
        if self.stationary:
            return

        prob_rotate = 0.3
        prob_start = 0.3
        prob_jump = 0.01
        prob_stop = 0.001

        # Rotate about a normal-distributed angle
        if self.y == 0 and random.uniform(0, 1) < prob_rotate:
            rot_y = self.rotation_y
            new_rot_y = rot_y + random.normalvariate(0, 180)
            self.animate_rotation((0, new_rot_y, 0), 1. / acceleration, curve=curve.linear)
        # Bounce off of other game objects
        hit_info = raycast(self.world_position, self.forward, 1, ignore=(self,))
        if hit_info.hit:
            self.speed = 0
            for _ in range(10):
                self.rotation_y += random.normalvariate(0, 180)
                hit_info = raycast(self.world_position, self.forward, 1, ignore=(self,))
                if not hit_info.hit:
                    break
        if self.y == 0 and self.speed == 0:
            if random.uniform(0, 1) < prob_start:
                # Start accelerating forward
                self.animate('speed', movement_speed, 1. / acceleration, curve=curve.linear)
        else:
            if self.y > 0:
                # Falling at 9.8 m/s^2
                self.speed_y -= time.dt * 9.8
            else:
                # Landing
                if self.y < 0:
                    self.y = 0
                    self.speed_y = 0
                # Jumping
                elif random.uniform(0, 1) < prob_jump:
                    self.speed_y = 0.5 * movement_speed
            if self.y == 0 and random.uniform(0, 1) < prob_stop:
                # Come to a halt
                self.animate('speed', 0, 1. / acceleration, curve=curve.linear)
        # Update horizontal and vertical positions
        self.position += self.forward * time.dt * self.speed
        self.y += time.dt * self.speed_y

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, value):
        self._hp = value
        if value <= 0:
            destroy(self)
            return

        self.health_bar.world_scale_x = self.hp / self.max_hp * 1.5
        self.health_bar.alpha = 1


enemies = []


def new_enemy():
    new = Enemy(
        x=random.uniform(-STAGE_X / 2 + WALL_THICKNESS, STAGE_X / 2 - WALL_THICKNESS),
        z=random.uniform(-STAGE_Z / 2 + WALL_THICKNESS, STAGE_Z / 2 - WALL_THICKNESS)
    )
    new.collider = BoxCollider(new, (0, 0.6, 0), (0.4, 1.2, 0.4))
    enemies.append(new)


for _ in range(NUM_ENEMIES):
    new_enemy()


def key_input(key):
    if key == 'tab':    # press tab to toggle edit/play mode
        editor_camera.enabled = not editor_camera.enabled

        player.visible_self = editor_camera.enabled
        player.cursor.enabled = not editor_camera.enabled
        gun.enabled = not editor_camera.enabled
        mouse.locked = not editor_camera.enabled
        editor_camera.position = player.position

        application.paused = editor_camera.enabled
    if key == 'right mouse down':
        ads(True)
    if key == 'right mouse up':
        ads(False)


input_handler = Entity(ignore_paused=True, input=key_input)


# sun = DirectionalLight()
# sun.look_at(Vec3(1,1,-1))
Sky()

app.run()
