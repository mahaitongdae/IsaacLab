import numpy as np
import onnxruntime as ort
onnx_model_path = '/home/haitong/PycharmProjects/google-research google-research master value_dice/save/value_dice/20240910-UnitreeGo2/160320-add_data_repr=0,algo=value_dice,env_name=UnitreeGo2,num_traj=1,repr_dim=512,seed=42-/policy_20000.onnx'
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
LegID = {
    "FR_0": 0,  # Front right hip
    "FR_1": 1,  # Front right thigh
    "FR_2": 2,  # Front right calf
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

def convertJointOrderIsaacToGo2(IsaacGymJoint):
    '''
    isaac gym: flhip, frhip, rlhip, rrhip,
                flthigh, frthigh, rlthigh, rrthigh,
                fl, fr, rl, rr calf
    go2: 0 for hip, 1 for thigh, 2 for calf
    '''
    Go2Joint = np.empty_like(IsaacGymJoint)
    Go2Joint[:,LegID["FL_0"]] = IsaacGymJoint[:,0]
    Go2Joint[:,LegID["FR_0"]] = IsaacGymJoint[:,1]
    Go2Joint[:,LegID["RL_0"]] = IsaacGymJoint[:,2]
    Go2Joint[:,LegID["RR_0"]] = IsaacGymJoint[:,3]
    Go2Joint[:,LegID["FL_1"]] = IsaacGymJoint[:,4]
    Go2Joint[:,LegID["FR_1"]] = IsaacGymJoint[:,5]
    Go2Joint[:,LegID["RL_1"]] = IsaacGymJoint[:,6]
    Go2Joint[:,LegID["RR_1"]] = IsaacGymJoint[:,7]
    Go2Joint[:,LegID["FL_2"]] = IsaacGymJoint[:,8]
    Go2Joint[:,LegID["FR_2"]] = IsaacGymJoint[:,9]
    Go2Joint[:,LegID["RL_2"]] = IsaacGymJoint[:,10]
    Go2Joint[:,LegID["RR_2"]] = IsaacGymJoint[:,11]
    return Go2Joint

def convertJointOrderGo2ToIsaac(Go2Joint):
    '''
    isaac gym: flhip, frhip, rlhip, rrhip,
                flthigh, frthigh, rlthigh, rrthigh,
                fl, fr, rl, rr calf
    go2: 0 for hip, 1 for thigh, 2 for calf
    '''
    IsaacGymJoint = np.empty_like(Go2Joint)
    IsaacGymJoint[:,0] = Go2Joint[:,LegID["FL_0"]]
    IsaacGymJoint[:,1] = Go2Joint[:,LegID["FR_0"]]
    IsaacGymJoint[:,2] = Go2Joint[:,LegID["RL_0"]]
    IsaacGymJoint[:,3] = Go2Joint[:,LegID["RR_0"]]
    IsaacGymJoint[:,4] = Go2Joint[:,LegID["FL_1"]]
    IsaacGymJoint[:,5] = Go2Joint[:,LegID["FR_1"]]
    IsaacGymJoint[:,6] = Go2Joint[:,LegID["RL_1"]]
    IsaacGymJoint[:,7] = Go2Joint[:,LegID["RR_1"]]
    IsaacGymJoint[:,8] = Go2Joint[:,LegID["FL_2"]]
    IsaacGymJoint[:,9] = Go2Joint[:,LegID["FR_2"]]
    IsaacGymJoint[:,10] = Go2Joint[:,LegID["RL_2"]]
    IsaacGymJoint[:,11] = Go2Joint[:,LegID["RR_2"]]
    return IsaacGymJoint

STAND = np.array([
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5,
        -0.0, 0.8, -1.5
    ])

def get_action(input_data):

    go2_input = np.empty_like(input_data)
    go2_input[:, :9] = input_data[:, :9]
    go2_input[:, 9:21] = convertJointOrderIsaacToGo2(input_data[:, 9:21]) - STAND
    go2_input[:, 21:33] = convertJointOrderIsaacToGo2(input_data[:, 21:33])
    if not isinstance(go2_input, np.ndarray):
        raise ValueError("Input data must be a numpy array")

    # Run the model on the input data
    output = session.run(None, {input_name: go2_input})

    # Return the results
    return output[0]