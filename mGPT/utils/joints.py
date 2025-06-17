# MyoSkeleton Joint Definitions - Complete replacement for SMPL
# Based on the anatomical structure from myoskeleton.xml

myoskeleton_joints = [
    # Root and pelvis
    "myoskeleton_root",
    "pelvis",
    
    # Lumbar spine (L5-L1)
    "lumbar_spine_5",
    "lumbar_spine_4", 
    "lumbar_spine_3",
    "lumbar_spine_2",
    "lumbar_spine_1",
    
    # Thoracic spine (consolidated)
    "thoracic_spine",
    
    # Cervical spine (C7-C1)
    "cervical_spine_7",
    "cervical_spine_6",
    "cervical_spine_5", 
    "cervical_spine_4",
    "cervical_spine_3",
    "cervical_spine_2",
    "cervical_spine_1",
    "skull",
    
    # Right arm chain
    "clavicle_r",
    "scapula_r", 
    "humerus_r",
    "ulna_r",
    "radius_r",
    "lunate_r",
    
    # Right hand (simplified - key joints)
    "metacarpal_1_r",  # thumb
    "proximal_phalanx_1_r",
    "distal_phalanx_1_r",
    "metacarpal_2_r",  # index
    "proximal_phalanx_2_r", 
    "intermediate_phalanx_2_r",
    "distal_phalanx_2_r",
    "metacarpal_3_r",  # middle
    "proximal_phalanx_3_r",
    "intermediate_phalanx_3_r", 
    "distal_phalanx_3_r",
    "metacarpal_4_r",  # ring
    "proximal_phalanx_4_r",
    "intermediate_phalanx_4_r",
    "distal_phalanx_4_r", 
    "metacarpal_5_r",  # pinky
    "proximal_phalanx_5_r",
    "intermediate_phalanx_5_r",
    "distal_phalanx_5_r",
    
    # Left arm chain (mirror of right)
    "clavicle_l",
    "scapula_l",
    "humerus_l", 
    "ulna_l",
    "radius_l",
    "lunate_l",
    
    # Left hand (simplified - key joints)
    "metacarpal_1_l",
    "proximal_phalanx_1_l",
    "distal_phalanx_1_l",
    "metacarpal_2_l",
    "proximal_phalanx_2_l",
    "intermediate_phalanx_2_l", 
    "distal_phalanx_2_l",
    "metacarpal_3_l",
    "proximal_phalanx_3_l",
    "intermediate_phalanx_3_l",
    "distal_phalanx_3_l",
    "metacarpal_4_l",
    "proximal_phalanx_4_l",
    "intermediate_phalanx_4_l",
    "distal_phalanx_4_l",
    "metacarpal_5_l", 
    "proximal_phalanx_5_l",
    "intermediate_phalanx_5_l",
    "distal_phalanx_5_l",
    
    # Right leg chain
    "hip_r",
    "femur_r", 
    "tibia_r",
    "fibula_r",
    "talus_r",
    "calcaneus_r",
    "foot_r",
    
    # Left leg chain  
    "hip_l",
    "femur_l",
    "tibia_l", 
    "fibula_l",
    "talus_l",
    "calcaneus_l",
    "foot_l",
]

# Simplified joint set for motion generation (key joints only)
myoskeleton_simplified_joints = [
    "myoskeleton_root",
    "pelvis",
    "lumbar_spine_3",  # mid lumbar
    "thoracic_spine", 
    "cervical_spine_4", # mid cervical
    "skull",
    
    # Arms (key joints)
    "clavicle_r", "humerus_r", "ulna_r", "lunate_r",
    "clavicle_l", "humerus_l", "ulna_l", "lunate_l", 
    
    # Hands (wrist + thumb + index)
    "proximal_phalanx_1_r", "proximal_phalanx_2_r",
    "proximal_phalanx_1_l", "proximal_phalanx_2_l",
    
    # Legs (key joints)
    "hip_r", "femur_r", "tibia_r", "foot_r",
    "hip_l", "femur_l", "tibia_l", "foot_l",
]

# Kinematic chains for MyoSkeleton
myoskeleton_kinematic_tree = [
    [0, 1, 2, 3, 4, 5],  # spine chain
    [5, 14, 15, 16, 17, 18, 19],  # right arm
    [5, 35, 36, 37, 38, 39, 40],  # left arm  
    [1, 56, 57, 58, 59, 60, 61, 62],  # right leg
    [1, 63, 64, 65, 66, 67, 68, 69],  # left leg
]

# Joint groups for constraints and analysis
myoskeleton_joints_info = {
    "root": myoskeleton_joints.index("myoskeleton_root"),
    "pelvis": myoskeleton_joints.index("pelvis"),
    "spine": [myoskeleton_joints.index(f"lumbar_spine_{i}") for i in range(5, 0, -1)] + 
             [myoskeleton_joints.index("thoracic_spine")] +
             [myoskeleton_joints.index(f"cervical_spine_{i}") for i in range(7, 0, -1)],
    "head": myoskeleton_joints.index("skull"),
    "shoulders": [myoskeleton_joints.index("clavicle_r"), myoskeleton_joints.index("clavicle_l")],
    "hands": [myoskeleton_joints.index("lunate_r"), myoskeleton_joints.index("lunate_l")],
    "hips": [myoskeleton_joints.index("hip_r"), myoskeleton_joints.index("hip_l")],
    "feet": [myoskeleton_joints.index("foot_r"), myoskeleton_joints.index("foot_l")],
}

# Motion constraints (for realistic motion generation)
myoskeleton_joint_limits = {
    # Spine flexion/extension limits (in radians)
    "lumbar_spine_5": {"flex": (-0.152, 0.044), "bend": (-0.06, 0.06), "arot": (-0.028, 0.028)},
    "lumbar_spine_4": {"flex": (-0.226, 0.065), "bend": (-0.08, 0.08), "arot": (-0.03, 0.03)}, 
    "lumbar_spine_3": {"flex": (-0.249, 0.071), "bend": (-0.108, 0.108), "arot": (-0.03, 0.03)},
    "lumbar_spine_2": {"flex": (-0.282, 0.081), "bend": (-0.11, 0.11), "arot": (-0.024, 0.024)},
    "lumbar_spine_1": {"flex": (-0.311, 0.089), "bend": (-0.083, 0.083), "arot": (-0.023, 0.023)},
    
    # Thoracic spine
    "thoracic_spine": {"flex": (-0.112, 0.036), "bend": (-0.061, 0.061), "arot": (-0.646, 0.646)},
    
    # Cervical spine ranges
    "cervical_spine_7": {"flex": (-0.057, 0.083), "bend": (-0.085, 0.085), "arot": (-0.063, 0.063)},
    "cervical_spine_1": {"flex": (-0.155, 0.233), "bend": (-0.052, 0.052), "arot": (-0.044, 0.044)},
}

def get_myoskeleton_root_idx():
    """Get the root joint index for MyoSkeleton"""
    return myoskeleton_joints_info["root"]

def get_simplified_joint_count():
    """Get the number of simplified joints for motion generation"""
    return len(myoskeleton_simplified_joints)

def is_spine_joint(joint_name):
    """Check if a joint is part of the spine chain"""
    return any(spine_part in joint_name for spine_part in 
               ["lumbar_spine", "thoracic_spine", "cervical_spine"])

def get_joint_limits(joint_name):
    """Get motion limits for a specific joint"""
    return myoskeleton_joint_limits.get(joint_name, None)
