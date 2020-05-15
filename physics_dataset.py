from pathlib import Path
from tqdm import tqdm
import random
import h5py
from typing import List, Dict
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.output_data import OutputData, Transforms, Rigidbodies, Collision, EnvironmentCollision, Images
import numpy as np


class _ObjectPosition:
    """
    Defines the initial position of an object.
    """

    def __init__(self, position: Dict[str, float], radius: float):
        """
        :param position: The position of the object.
        :param radius: The maximum radius swept by the object's bounds.
        """

        self.position = position
        self.radius = radius


class PhysicsDataset(Controller):
    """
    Per trial, create 2-3 "toys". Apply a force to one of them, directed at another.
    Per frame, save object/physics metadata and image data.
    """

    def __init__(self, port: int = 1071):
        lib = ModelLibrarian(str(Path("toys.json").resolve()))
        self.records = lib.records

        super().__init__(port=port)

    def run(self, num: int, output_dir: str) -> None:
        """
        Create the dataset.

        :param num: The number of trials in the dataset.
        :param output_dir: The root output directory.
        """

        pbar = tqdm(total=num)
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Initialize the scene.
        self.communicate([{"$type": "set_screen_size",
                           "width": 256,
                           "height": 256},
                          {"$type": "set_render_quality",
                           "render_quality": 5},
                          {"$type": "set_physics_solver_iterations",
                           "iterations": 32},
                          self.get_add_scene(scene_name="box_room_2018"),
                          {"$type": "set_aperture",
                           "aperture": 4.8},
                          {"$type": "set_focus_distance",
                           "focus_distance": 1.25},
                          {"$type": "set_post_exposure",
                           "post_exposure": 0.4},
                          {"$type": "set_vignette",
                           "enabled": False},
                          {"$type": "set_ambient_occlusion_intensity",
                           "intensity": 0.175},
                          {"$type": "set_ambient_occlusion_thickness_modifier",
                           "thickness": 3.5},
                          {"$type": "set_sleep_threshold",
                           "sleep_threshold": 0.1},
                          {"$type": "set_shadow_strength",
                           "strength": 1.0},
                          {"$type": "create_avatar",
                           "type": "A_Img_Caps_Kinematic",
                           "id": "a"},
                          {"$type": "set_pass_masks",
                           "pass_masks": ["_img", "_id", "_depth", "_normals"]},
                          {"$type": "set_field_of_view",
                           "field_of_view": 55},
                          {"$type": "send_images",
                           "frequency": "always"}])

        for i in range(num):
            filepath = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + ".hdf5")
            # If the file already exists, check to see if it is corrupted. If it is, remove the file.
            # If it is ok, assume that the dataset was completed, and skip it.
            if filepath.exists():
                try:
                    h5py.File(str(filepath.resolve()), "r")
                except OSError:
                    filepath.unlink()
            if not filepath.exists():
                # Do the trial.
                self.trial(filepath=filepath)
            pbar.update(1)
        pbar.close()
        self.communicate({"$type": "terminate"})

    def trial(self, filepath: Path) -> None:
        """
        Create 2-3 objects. Set random physics parameters, camera position, etc.
        Apply a force to one object, directed at another.
        Per frame, write object and image data to disk.

        :param filepath: The path to this trial's hdf5 file.
        """
        # Start a new file.
        f = h5py.File(str(filepath.resolve()), "a")

        # Positions where objects will be placed (used to prevent interpenetration).
        object_positions: List[_ObjectPosition] = []

        num_objects = random.choice([2, 3])

        # Static data for this trial.
        object_ids = np.empty(dtype=int, shape=(0, num_objects, 1))
        masses = np.empty(dtype=np.float32, shape=(0, num_objects, 1))
        static_frictions = np.empty(dtype=np.float32, shape=(0, num_objects, 1))
        dynamic_frictions = np.empty(dtype=np.float32, shape=(0, num_objects, 1))
        bouncinesses = np.empty(dtype=np.float32, shape=(0, num_objects, 1))

        # Randomize the order of the records and pick the first one.
        # This way, the objects are always different.
        random.shuffle(self.records)

        commands = []

        # Add 2-3 objects.
        for i in range(num_objects):
            o_id = Controller.get_unique_id()
            record = self.records[i]

            # Set randomized physics values and update the physics info.
            scale = TDWUtils.get_unit_scale(record) * random.uniform(0.8, 1.1)

            # Get a random position.
            o_pos = self._get_object_position(object_positions=object_positions)
            # Add the object and the radius, which is defined by its scale.
            object_positions.append(_ObjectPosition(position=o_pos, radius=scale))

            # Set random physics properties.
            mass = random.uniform(1, 5)
            dynamic_friction = random.uniform(0, 0.9)
            static_friction = random.uniform(0, 0.9)
            bounciness = random.uniform(0, 1)

            # Log the static data.
            object_ids = np.append(object_ids, o_id)
            masses = np.append(masses, mass)
            dynamic_frictions = np.append(dynamic_frictions, dynamic_friction)
            static_frictions = np.append(static_frictions, static_friction)
            bouncinesses = np.append(bouncinesses, bounciness)

            # Add commands to create the object and set its physics values.
            commands.extend([{"$type": "add_object",
                              "id": o_id,
                              "name": record.name,
                              "url": record.get_url(),
                              "position": o_pos,
                              "rotation": {"x": 0, "y": random.uniform(-90, 90), "z": 0},
                              "scale_factor": record.scale_factor,
                              "category": record.wcategory},
                             {"$type": "scale_object",
                              "id": o_id,
                              "scale_factor": {"x": scale, "y": scale, "z": scale}},
                             {"$type": "set_mass",
                              "id": o_id,
                              "mass": mass},
                             {"$type": "set_physic_material",
                              "id": o_id,
                              "dynamic_friction": dynamic_friction,
                              "static_friction": static_friction,
                              "bounciness": bounciness},
                             {"$type": "set_object_collision_detection_mode",
                              "id": o_id,
                              "mode": "continuous_dynamic"}])
        # Get a random position for the avatar.
        # Offset the initial position of the avatar from the center of the room.
        a_r = random.uniform(0.9, 1.5)
        a_x = a_r
        a_y = random.uniform(0.5, 1.25)
        a_z = a_r
        theta = np.radians(random.uniform(0, 360))
        a_x = np.cos(theta) * a_x - np.sin(theta) * a_z
        a_z = np.sin(theta) * a_x + np.cos(theta) * a_z
        a_pos = {"x": a_x, "y": a_y, "z": a_z}

        # Point one object at the center, and then offset the rotation.
        # Apply a force allow the forward directional vector.
        # Teleport the avatar and look at the object that will be hit. Then slightly rotate the camera randomly.
        # Listen for output data.
        force_id = int(object_ids[0])
        target_id = int(object_ids[1])
        commands.extend([{"$type": "object_look_at",
                          "other_object_id": target_id,
                          "id": force_id},
                         {"$type": "rotate_object_by",
                          "angle": random.uniform(-5, 5),
                          "id": force_id,
                          "axis": "yaw",
                          "is_world": True},
                         {"$type": "apply_force_magnitude_to_object",
                          "magnitude": random.uniform(20, 60),
                          "id": force_id},
                         {"$type": "teleport_avatar_to",
                          "position": a_pos},
                         {"$type": "look_at",
                          "object_id": target_id,
                          "use_centroid": True},
                         {"$type": "rotate_sensor_container_by",
                          "axis": "pitch",
                          "angle": random.uniform(-5, 5)},
                         {"$type": "rotate_sensor_container_by",
                          "axis": "yaw",
                          "angle": random.uniform(-5, 5)},
                         {"$type": "focus_on_object",
                          "object_id": target_id},
                         {"$type": "send_transforms",
                          "frequency": "always"},
                         {"$type": "send_collisions",
                          "enter": True,
                          "exit": False,
                          "stay": False,
                          "collision_types": ["obj", "env"]},
                         {"$type": "send_rigidbodies",
                          "frequency": "always"}])

        # Write the static data to the disk.
        static_group = f.create_group("static")
        static_group.create_dataset("object_ids", data=object_ids)
        static_group.create_dataset("mass", data=masses)
        static_group.create_dataset("static_friction", data=static_frictions)
        static_group.create_dataset("dynamic_friction", data=dynamic_frictions)
        static_group.create_dataset("bounciness", data=bouncinesses)

        # Send the commands and start the trial.
        resp = self.communicate(commands)
        frame = 0

        frames_grp = f.create_group("frames")

        # Add the first frame.
        done = False
        self._add_frame(grp=frames_grp, resp=resp, frame_num=frame, object_ids=object_ids)

        # Continue the trial. Send commands, and parse output data.
        while not done and frame < 1000:
            frame += 1
            resp = self.communicate({"$type": "focus_on_object",
                                     "object_id": target_id})
            done = self._add_frame(grp=frames_grp, resp=resp, frame_num=frame, object_ids=object_ids)

        # Cleanup.
        commands = []
        for o_id in object_ids:
            commands.append({"$type": "destroy_object",
                             "id": int(o_id)})
        self.communicate(commands)
        f.close()

    @staticmethod
    def _get_object_position(object_positions: List[_ObjectPosition], max_tries: int = 1000, radius: float = 2) -> \
            Dict[str, float]:
        """
        Try to get a valid random position that doesn't interpentrate with other objects.

        :param object_positions: The positions and radii of all objects so far that will be added to the scene.
        :param max_tries: Try this many times to get a valid position before giving up.
        :param radius: The radius to pick a position in.

        :return: A valid position that doesn't interpentrate with other objects.
        """

        o_pos = TDWUtils.array_to_vector3(TDWUtils.get_random_point_in_circle(center=np.array([0, 0, 0]),
                                                                              radius=radius))
        # Pick a position away from other objects.
        ok = False
        count = 0
        while not ok and count < max_tries:
            count += 1
            ok = True
            for o in object_positions:
                # If the object is too close to another object, try another position.
                if TDWUtils.get_distance(o.position, o_pos) <= o.radius:
                    ok = False
                    o_pos = TDWUtils.array_to_vector3(TDWUtils.get_random_point_in_circle(center=np.array([0, 0, 0]),
                                                                                          radius=radius))
        return o_pos

    @staticmethod
    def _add_frame(grp: h5py.Group, resp: List[bytes], frame_num: int, object_ids: np.array) -> bool:
        """
        Add a frame to the hdf5 file.

        :param grp: The hd5f group.
        :param resp: The response from the build.
        :param frame_num: The frame number.
        :param object_ids: An ordered list of object IDs.

        :return: True if all objects are not moving (or close to not moving).
        """

        num_objects = len(object_ids)

        frame = grp.create_group(TDWUtils.zero_padding(frame_num, 4))
        images = frame.create_group("images")

        # Transforms data.
        positions = np.empty(dtype=np.float32, shape=(num_objects, 3))
        forwards = np.empty(dtype=np.float32, shape=(num_objects, 3))
        rotations = np.empty(dtype=np.float32, shape=(num_objects, 4))
        # Physics data.
        velocities = np.empty(dtype=np.float32, shape=(num_objects, 3))
        angular_velocities = np.empty(dtype=np.float32, shape=(num_objects, 3))
        # Collision data.
        collision_ids = np.empty(dtype=np.int32, shape=(0, 2))
        collision_relative_velocities = np.empty(dtype=np.float32, shape=(0, 3))
        collision_contacts = np.empty(dtype=np.float32, shape=(0, 2, 3))
        # Environment Collision data.
        env_collision_ids = np.empty(dtype=np.int32, shape=(0, 1))
        env_collision_contacts = np.empty(dtype=np.float32, shape=(0, 2, 3))

        # This is set to False if any objects aren't sleeping.
        sleeping = True
        # A list of all objects below the floor; they are ignored when checking for sleeping objects.
        in_abyss: List[int] = []

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "tran":
                tr = Transforms(r)
                # Parse the data in an ordered manner so that it can be mapped back to the object IDs.
                tr_dict = dict()
                for i in range(tr.get_num()):
                    pos = tr.get_position(i)
                    # If an object has fallen into the abyss, ignore it when checking for sleeping objects.
                    if pos[1] <= -1:
                        in_abyss.append(tr.get_id(i))
                    tr_dict.update({tr.get_id(i): {"pos": pos,
                                                   "for": tr.get_forward(i),
                                                   "rot": tr.get_rotation(i)}})
                # Add the Transforms data.
                for o_id, i in zip(object_ids, range(num_objects)):
                    positions[i] = tr_dict[o_id]["pos"]
                    forwards[i] = tr_dict[o_id]["for"]
                    rotations[i] = tr_dict[o_id]["rot"]
            elif r_id == "rigi":
                ri = Rigidbodies(r)
                ri_dict = dict()
                for i in range(ri.get_num()):
                    ri_dict.update({ri.get_id(i): {"vel": ri.get_velocity(i),
                                                   "ang": ri.get_angular_velocity(i)}})
                    # Check if any objects are sleeping that aren't in the abyss.
                    if not ri.get_sleeping(i) and ri.get_id(i) not in in_abyss:
                        sleeping = False
                # Add the Rigibodies data.
                for o_id, i in zip(object_ids, range(num_objects)):
                    velocities[i] = ri_dict[o_id]["vel"]
                    angular_velocities[i] = ri_dict[o_id]["ang"]
            elif r_id == "imag":
                im = Images(r)
                # Add each image.
                for i in range(im.get_num_passes()):
                    images.create_dataset(im.get_pass_mask(i), data=im.get_image(i), compression="gzip")
            elif r_id == "coll":
                co = Collision(r)
                collision_ids = np.append(collision_ids, [co.get_collider_id(), co.get_collidee_id()])
                collision_relative_velocities = np.append(collision_relative_velocities, co.get_relative_velocity())
                for i in range(co.get_num_contacts()):
                    collision_contacts = np.append(collision_contacts, (co.get_contact_normal(i),
                                                                        co.get_contact_point(i)))
            elif r_id == "enco":
                en = EnvironmentCollision(r)
                env_collision_ids = np.append(env_collision_ids, en.get_object_id())
                for i in range(en.get_num_contacts()):
                    env_collision_contacts = np.append(env_collision_contacts, (en.get_contact_normal(i),
                                                                                en.get_contact_point(i)))

        # Write the data to disk.
        objs = frame.create_group("objects")
        objs.create_dataset("positions", data=positions.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("forwards", data=forwards.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("rotations", data=rotations.reshape(num_objects, 4), compression="gzip")
        objs.create_dataset("velocities", data=velocities.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("angular_velocities", data=angular_velocities.reshape(num_objects, 3), compression="gzip")
        collisions = frame.create_group("collisions")
        collisions.create_dataset("object_ids", data=collision_ids.reshape((-1, 2)), compression="gzip")
        collisions.create_dataset("relative_velocities", data=collision_relative_velocities.reshape((-1, 3)),
                                  compression="gzip")
        collisions.create_dataset("contacts", data=collision_contacts.reshape((-1, 2, 3)), compression="gzip")
        env_collisions = frame.create_group("env_collisions")
        env_collisions.create_dataset("object_ids", data=env_collision_ids, compression="gzip")
        env_collisions.create_dataset("contacts", data=env_collision_contacts.reshape((-1, 2, 3)),
                                      compression="gzip")
        return sleeping


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="D:/physics_dataset", help="Root output directory.")
    parser.add_argument("--num", type=int, default=3000, help="The number of trials in the dataset.")

    args = parser.parse_args()
    p = PhysicsDataset()
    p.run(num=args.num, output_dir=args.dir)
