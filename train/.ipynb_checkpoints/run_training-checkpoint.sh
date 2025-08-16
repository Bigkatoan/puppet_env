#!/bin/bash

# ==============================================================================
#  Training Script for End-to-End Autonomous Navigation
#  This script automates the two-phase training process described in the
#  research documentation.
# ==============================================================================

# --- Color Definitions for Readable Output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Function to Print Usage Instructions ---
print_usage() {
    echo -e "\n${YELLOW}Usage: $0 [phase] [agent_type]${NC}"
    echo ""
    echo "  ${GREEN}Phases:${NC}"
    echo "    phase1          - Runs the complete pre-training for both perception models."
    echo "    phase2          - Runs the RL agent training using pre-trained models."
    echo ""
    echo "  ${GREEN}Agent Types (required for phase2):${NC}"
    echo "    sac             - Train the Soft Actor-Critic agent."
    echo "    ddpg            - Train the Deep Deterministic Policy Gradient agent."
    echo "    ppo             - Train the Proximal Policy Optimization agent."
    echo ""
    echo -e "  ${GREEN}Examples:${NC}"
    echo "    ./run_training.sh phase1"
    echo "    ./run_training.sh phase2 sac"
    echo "    ./run_training.sh phase2 ppo"
    echo ""
}

# --- Check for correct number of arguments ---
if [ "$#" -eq 0 ]; then
    print_usage
    exit 1
fi

PHASE=$1
AGENT=$2

# =========================================
#  PHASE 1: PERCEPTION MODEL PRE-TRAINING
# =========================================
if [ "$PHASE" == "phase1" ]; then
    echo -e "${YELLOW}--- Starting Phase 1: Pre-training Perception Models ---${NC}"

    # --- Step 1.1: Train Edge-Autoencoder ---
    echo -e "\n${GREEN}Step 1.1: Training Edge-Autoencoder...${NC}"
    # This command uses the parameters from your documentation (Table 7)
    python train_phase1.py --model autoencoder --epochs 10 --batch-size 32

    # Check if training succeeded before proceeding
    if [ $? -ne 0 ]; then
        echo -e "\n${RED}Error: Edge-Autoencoder training failed. Aborting.${NC}"
        exit 1
    fi
    if [ ! -f "edge_autoencoder.pth" ]; then
        echo -e "\n${RED}Error: edge_autoencoder.pth not found after training. Aborting.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Edge-Autoencoder training complete! Model saved as edge_autoencoder.pth.${NC}"

    # --- Step 1.2: Train Pose Prediction Network ---
    echo -e "\n${GREEN}Step 1.2: Training Pose Prediction Network...${NC}"
    python train_phase1.py --model posenet --epochs 15 --batch-size 32

    if [ $? -ne 0 ]; then
        echo -e "\n${RED}Error: Pose Prediction Network training failed. Aborting.${NC}"
        exit 1
    fi
    if [ ! -f "pose_network.pth" ]; then
        echo -e "\n${RED}Error: pose_network.pth not found after training. Aborting.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Pose Prediction Network training complete! Model saved as pose_network.pth.${NC}"
    echo -e "\n${YELLOW}--- Phase 1 Complete! You are now ready to train the RL agents. ---${NC}"

# =========================================
#  PHASE 2: RL AGENT TRAINING
# =========================================
elif [ "$PHASE" == "phase2" ]; then
    # Check if an agent type was specified
    if [ -z "$AGENT" ]; then
        echo -e "\n${RED}Error: You must specify an agent type for Phase 2 (sac, ddpg, or ppo).${NC}"
        print_usage
        exit 1
    fi

    # Check if perception models exist before starting
    if [ ! -f "edge_autoencoder.pth" ] || [ ! -f "pose_network.pth" ]; then
        echo -e "\n${RED}Error: Pre-trained models not found!${NC}"
        echo -e "${YELLOW}Please run './run_training.sh phase1' first to generate them.${NC}"
        exit 1
    fi
    
    echo -e "\n${YELLOW}--- Starting Phase 2: Training RL Agent ($AGENT) ---${NC}"
    # This command calls the main RL training script with the specified agent
    python train.py --agent "$AGENT"

    if [ $? -ne 0 ]; then
        echo -e "\n${RED}Error: RL training for agent '$AGENT' failed.${NC}"
        exit 1
    fi
    echo -e "\n${YELLOW}--- Phase 2 ($AGENT) Complete! Check the 'logs' and 'trained_models' directories. ---${NC}"

# =========================================
#  INVALID USAGE
# =========================================
else
    echo -e "\n${RED}Error: Invalid phase '$PHASE'. Choose 'phase1' or 'phase2'.${NC}"
    print_usage
    exit 1
fi