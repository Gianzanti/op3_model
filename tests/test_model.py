import os
import time
import unittest

import mujoco
import mujoco.viewer
import numpy as np
import imageio
# from pyvirtualdisplay import Display

# Start a virtual display
# vdisplay = Display(visible=False, size=(400, 400))
# vdisplay.start()

model_path = os.path.join(os.path.dirname(__file__), "..", "src", "model", "scene.xml")

class DarwinOp3_TestModel(unittest.TestCase):
    def test_model_properties(self):
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        print ("Model properties")
        print ("nq (number of generalized coordinates = dim(qpos)): ", model.nq)
        print ("nv (number of degrees of freedom = dim(qvel)): ", model.nv)
        print ("nu (number of actuators/controls = dim(ctrl)): ", model.nu)
        print ("ngeom (number of geometric elements): ", model.ngeom)
        print ("nbody (number of bodies): ", model.nbody)
        print ("njnt (number of joints): ", model.njnt)
        print ("nsensor (number of sensors): ", model.nsensor)
        print ("nsensordata: ", model.nsensordata)
        print ("Data properties")
        print ("qpos: ", data.qpos)
        print ("qvel: ", data.qvel)
        print ("qacc: ", data.qacc)
        print ("sensor_data: ", data.sensordata)
        print ("Gyroscope: ", np.array([data.sensordata[0:3]]))
        print ("Accelerometer: ", np.array([data.sensordata[3:6]]))
        print ("Magnetometer: ", np.array([data.sensordata[6:9]]))

        for i in range(model.njnt):
            print (f"Joint {i}: name={model.joint(i).name}, type={model.joint(i).type}, pos={data.qpos[i]}, range={model.jnt_range[i]}")

        for i in range(model.nbody):
            print (f"Body {i}: name={model.body(i).name}, parent={model.body(i).parentid}")

        for i in range(model.ngeom):
            print (f"Geom {i}: name={model.geom(i).name}, type={model.geom(i).type}, pos={data.geom_xpos[i]}")

    def test_viewer(self):
        model = mujoco.MjModel.from_xml_path(model_path)
        self.assertIsNotNone(model)

        data = mujoco.MjData(model)
        self.assertIsNotNone(data)

        # set initial state
        mujoco.mj_resetData(model, data)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()

            while viewer.is_running() and time.time() - start < 600:
                step_start = time.time()

                # print (f"Gyr: {np.array(data.sensordata[0:3])}")
                # print (f"Acc: {np.array(data.sensordata[3:6])}")
                # print (f"Mag: {np.array(data.sensordata[6:9])}")

                # mujoco.mjv_initGeom(
                #     viewer.user_scn.geoms[viewer.user_scn.ngeom],
                #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
                #     size=[0.02, 0.02, 0.02],
                #     pos=pos,
                #     mat=None,
                #     rgba=[1, 0, 0, 1]
                # )
                # viewer.user_scn.ngeom += 1

                lfoot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot")
                pos = data.geom_xpos[lfoot_geom_id]  # np.array([x, y, z])
                print("Left foot geom position:", pos)

                rfoot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot")
                pos = data.geom_xpos[rfoot_geom_id]  # np.array([x, y, z])
                print("Right foot geom position:", pos)


                lknee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee")
                angvel = data.qvel[lknee_joint_id]
                print("Left knee joint angular velocity (rad/s):", angvel)

                rknee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_knee")
                angvel = data.qvel[rknee_joint_id]
                print("Right knee joint angular velocity (rad/s):", angvel)

                mujoco.mj_step(model, data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)



        