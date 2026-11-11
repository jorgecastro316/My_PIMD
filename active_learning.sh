#!/usr/bin/env bash

#SBATCH -J ALFE
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
##SBATCH --gres=gpu:1
#SBATCH -o job_%A_%a.out
#SBATCH -e job_%A_%a.err
#SBATCH --time=02-00:00:00
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Set VENV_PATH when the required packages are installed in a virtual
# environment that is not already active.
if [[ -n "${VENV_PATH:-}" ]]; then
    source "$VENV_PATH/bin/activate"
fi

NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"

# === PARAMETERS ===
NBEADS=8
BETA=1
TOL=0.01
MAX_ROUNDS=500
NPOOL=10000
NPIMC=10
NSTEPS=50
NTRAIN=1000
TOPK=100
UNC_THRESHOLD=0.01  # uncertainty threshold for selecting low-unc points
LO_SUBSET=500
LO_METHOD=fps
AUTO_UNC_FACTOR=0.2
TRAIN_CAP=250
ROUND=0
NITER=200        # gpytorch training steps
LR=0.05          # learning rate for ADAM
NOISE_MIN=1e-5   # minimum noise floor (stabilizes GP)
JITTER=1e-3
CG_MAX=2000
PRECOND=200
FLOOD_THRESH=5.0

SEED_FORWARD="${SEED_FORWARD:-seed/mother_F}"
SEED_REVERSE="${SEED_REVERSE:-seed/mother_R}"
INPUT_FORWARD="${INPUT_FORWARD:-examples/active_learning/input_forward.txt}"
INPUT_REVERSE="${INPUT_REVERSE:-examples/active_learning/input_reverse.txt}"

mkdir -p logs checkpoints data/bar_seed data/gpr_seed data/candidates model plots
global_start=$(date +%s)
# === INIT ROUND 0 ===
if [ ! -f data/bar_seed/round_0.npz ]; then
    echo "[INIT] Building BAR seed (round 0)"
    python3 controller/controller.py "$SEED_FORWARD" "$SEED_REVERSE" "$BETA" \
        --prefix-work simulation \
        --out data/bar_seed/round_0.npz \
        --ndata $NTRAIN \
        --beads
    cp data/bar_seed/round_0.npz data/gpr_seed/round_0.npz
fi

if [ ! -f checkpoints/round_0.gpr.done ]; then
    echo "[INIT] Training initial GPR model"
    python3 controller/controller.py \
        --gpr-train data/gpr_seed/round_0.npz \
        --gpr-model model/round_0 \
        --n-iter $NITER \
	    --ard True \
        --lr $LR \
        --noise-min $NOISE_MIN \
        --feature-mode nmodes:all \
        --jitter $JITTER \
        --cg-max-iter $CG_MAX \
        --precond-size $PRECOND

    touch checkpoints/round_0.gpr.done
fi

# === LOOP ===
while [ $ROUND -lt $MAX_ROUNDS ]; do
    NEXT=$((ROUND+1))
    echo
    echo "======================"
    echo "[ ROUND $ROUND → $NEXT ]"
    echo "======================"
    round_start=$(date +%s)
    mkdir -p data/candidates/round_${ROUND}

    # 1. Generate pools
    if [ ! -f checkpoints/round_${ROUND}.pools.done ]; then
        python3 controller/controller.py --pimc-fps-pool data/candidates/round_${ROUND}/forward_pool.npz \
            --N $NPOOL --X $NPIMC --nbeads $NBEADS --direction forward --enhanced --gen-vel \
            --no-text --input-file "$INPUT_FORWARD" \
            --seed-npz data/gpr_seed/round_${ROUND}.npz

        python3 controller/controller.py --pimc-fps-pool data/candidates/round_${ROUND}/reverse_pool.npz \
            --N $NPOOL --X $NPIMC --nbeads $NBEADS --direction reverse --enhanced --gen-vel \
            --no-text --input-file "$INPUT_REVERSE" \
            --seed-npz data/gpr_seed/round_${ROUND}.npz
        touch checkpoints/round_${ROUND}.pools.done
    fi


    # 2. Predict
    if [ -f model/round_${ROUND}/train_meta.npz ]; then
        used_round=$(python3 -c "import numpy as np; d=np.load('model/round_${ROUND}/train_meta.npz'); print(d['nF']+d['nR'])" 2>/dev/null || echo "unknown")
        echo "[INFO] Using GPR model in model/round_${ROUND}/ (trained on ${used_round} points)"
    else
        echo "[WARN] No model metadata found in model/round_${ROUND}/"
    fi

    if [ ! -f checkpoints/round_${ROUND}.predict.done ]; then
        python3 controller/controller.py --gpr-predict-both \
            --cands-forward data/candidates/round_${ROUND}/forward_pool.npz \
            --cands-reverse data/candidates/round_${ROUND}/reverse_pool.npz \
            --gpr-model model/round_${ROUND}/ \
            --select-out-forward data/candidates/round_${ROUND}/idx_hi_forward.npy \
            --select-keep-forward data/candidates/round_${ROUND}/idx_lo_forward.npy \
            --select-out-reverse data/candidates/round_${ROUND}/idx_hi_reverse.npy \
            --select-keep-reverse data/candidates/round_${ROUND}/idx_lo_reverse.npy \
	        --unc-threshold $UNC_THRESHOLD \
            --topk $TOPK \
            --lo-subset-forward $LO_SUBSET \
            --lo-subset-reverse $LO_SUBSET \
            --lo-subset-method $LO_METHOD \
            --lo-subset-seed $ROUND \
            --auto-unc-factor-forward $AUTO_UNC_FACTOR \
            --auto-unc-factor-reverse $AUTO_UNC_FACTOR
        touch checkpoints/round_${ROUND}.predict.done
    fi

    # 3. Run SPIMD
    if [ ! -f checkpoints/round_${ROUND}.spimd.done ]; then
        python3 controller/controller.py --run-spimd-from-pool \
            --pool-npz data/candidates/round_${ROUND}/forward_pool.npz \
            --select-indices data/candidates/round_${ROUND}/idx_hi_forward.npy \
            --direction forward --nsteps $NSTEPS --input-file "$INPUT_FORWARD" \
            --out data/candidates/round_${ROUND}/spimd_forward.npz

        python3 controller/controller.py --run-spimd-from-pool \
            --pool-npz data/candidates/round_${ROUND}/reverse_pool.npz \
            --select-indices data/candidates/round_${ROUND}/idx_hi_reverse.npy \
            --direction reverse --nsteps $NSTEPS --input-file "$INPUT_REVERSE" \
            --out data/candidates/round_${ROUND}/spimd_reverse.npz
        touch checkpoints/round_${ROUND}.spimd.done
    fi

    # 4. Merge BAR
    if [ ! -f checkpoints/round_${NEXT}.bar_merge.done ]; then
        python3 controller/controller.py --merge-data \
            --seed-old data/bar_seed/round_${ROUND}.npz \
            --add-forward data/candidates/round_${ROUND}/predicted_lo_forward.npz data/candidates/round_${ROUND}/spimd_forward.npz \
            --add-reverse data/candidates/round_${ROUND}/predicted_lo_reverse.npz data/candidates/round_${ROUND}/spimd_reverse.npz \
            --out data/bar_seed/round_${NEXT}.npz \
            --flood-threshold $FLOOD_THRESH \
            --train-select $TRAIN_CAP
        touch checkpoints/round_${NEXT}.bar_merge.done
    fi

    # 5. Merge GPR
    if [ ! -f checkpoints/round_${NEXT}.gpr_merge.done ]; then
        python3 controller/controller.py --merge-data \
            --seed-old data/gpr_seed/round_${ROUND}.npz \
            --add-forward data/candidates/round_${ROUND}/spimd_forward.npz \
            --add-reverse data/candidates/round_${ROUND}/spimd_reverse.npz \
            --out data/gpr_seed/round_${NEXT}.npz \
            --flood-threshold $FLOOD_THRESH \
            --train-select $TRAIN_CAP
        touch checkpoints/round_${NEXT}.gpr_merge.done
    fi

    # 6. Compute ΔF
    if [ ! -f checkpoints/round_${NEXT}.bar_done ]; then
        BAR_OUTPUT=$(python3 controller/controller.py --bar-from-seed data/bar_seed/round_${NEXT}.npz --beta $BETA)
        echo "$BAR_OUTPUT"

        DF=$(echo "$BAR_OUTPUT" | grep "ΔF" | sed -E 's/.*ΔF = ([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?).*/\1/')
        WIDTH=$(echo "$BAR_OUTPUT" | grep "width=" | sed -E 's/.*width=([0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?).*/\1/')

        echo "[ROUND $NEXT] ΔF = $DF | CI width = $WIDTH"

        if [ -z "$WIDTH" ]; then
            echo "[ERROR] Could not parse CI width, aborting"
            exit 1
        fi

        if (( $(echo "$WIDTH < $TOL" | bc -l) )); then
            echo "[STOP] Converged at round $NEXT with CI width = $WIDTH"
            exit 0
        fi

        touch checkpoints/round_${NEXT}.bar_done
    fi

    # 7. Train GPR
    if [ ! -f checkpoints/round_${NEXT}.gpr_train.done ]; then
        python3 controller/controller.py \
            --gpr-train data/gpr_seed/round_${NEXT}.npz \
            --gpr-model model/round_${NEXT} \
            --n-iter $NITER \
            --lr $LR \
            --ard True \
            --noise-min $NOISE_MIN \
            --feature-mode nmodes:all \
            --jitter $JITTER \
            --cg-max-iter $CG_MAX \
            --precond-size $PRECOND

        touch checkpoints/round_${NEXT}.gpr_train.done
        nbar=$(python3 -c "import numpy as np; d=np.load('data/bar_seed/round_${NEXT}.npz'); print(len(d['qF'])+len(d['qR']))")
        ngpr=$(python3 -c "import numpy as np; d=np.load('data/gpr_seed/round_${NEXT}.npz'); print(len(d['qF'])+len(d['qR']))")
        nmodel=$(python3 -c "import os, numpy as np; f='model/round_${NEXT}/train_meta.npz'; \
        print(int(np.load(f)['nF'])+int(np.load(f)['nR']) if os.path.exists(f) else -1)")
        echo "[SUMMARY] Round $NEXT: bar_seed=$nbar, gpr_seed=$ngpr, model_trained_on=$nmodel"

    fi

    ROUND=$NEXT
    # --- Print round runtime ---
    round_end=$(date +%s)
    round_elapsed=$((round_end - round_start))
    printf "[TIMER] Round %d completed in %02dh:%02dm:%02ds\n" \
           "$ROUND" $((round_elapsed/3600)) $((round_elapsed%3600/60)) $((round_elapsed%60))

done

# --- Print total runtime ---
global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
printf "[TIMER] Total process finished in %02dh:%02dm:%02ds\n" \
       $((global_elapsed/3600)) $((global_elapsed%3600/60)) $((global_elapsed%60))

echo "[WARN] Reached MAX_ROUNDS=$MAX_ROUNDS without convergence"
