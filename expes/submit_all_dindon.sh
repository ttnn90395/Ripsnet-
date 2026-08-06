#!/bin/bash
# Submit ALL TFN model ablation experiments on dindon cluster
# Uses all available nodes via large job arrays

BASE="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"
RESULTS="$BASE/results/ablations"
mkdir -p "$RESULTS"

DATASETS=(
  CBF ChlorineConcentration DistalPhalanxOutlineCorrect
  ECG200 ECG5000 GunPoint GunPointOldVersusYoung
  ItalyPowerDemand MedicalImages MiddlePhalanxOutlineAgeGroup
  MiddlePhalanxOutlineCorrect MiddlePhalanxTW
  PhalangesOutlinesCorrect Plane PowerCons
  ProximalPhalanxOutlineAgeGroup ProximalPhalanxTW
  SonyAIBORobotSurface1 SonyAIBORobotSurface2
  TwoLeadECG UMD
)

MODELS=(
  TensorFieldNetwork GTTensorFieldNetwork GTTensorFieldNetworkV2
  HierarchicalGTTFN HierarchicalTensorFieldNetwork
  OnEquivariantTensorFieldNetwork AttentionTensorFieldNetwork
  StochasticTensorFieldNetwork CrossAttentionTensorFieldNetwork
  RelaxedOnEquivariantTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork
  PointNetTutorial PointNet3D DistanceMatrixRaggedModel
  ScalarDistanceDeepSet ScalarInputMLP MultiInputModel
  RaggedPersistenceModel
)

PERCENTAGES=(10 20 30 50 70 100)
TRIALS=(0 1 2 3)
EPOCHS=100
IDENT="try1"

# Build flat index mapping
# idx = di*MLPT + mi*PT + pi*T + ti
ND=${#DATASETS[@]}
NM=${#MODELS[@]}
NP=${#PERCENTAGES[@]}
NT=${#TRIALS[@]}
TOTAL=$((ND * NM * NP * NT))

echo "Submitting $TOTAL jobs ($ND datasets x $NM models x $NP percentages x $NT trials)"

# Write parameter map file
PARAM_FILE="$RESULTS/param_map.txt"
> "$PARAM_FILE"
idx=0
for di in $(seq 0 $((ND-1))); do
  for mi in $(seq 0 $((NM-1))); do
    for pi in $(seq 0 $((NP-1))); do
      for ti in $(seq 0 $((NT-1))); do
        echo "$idx ${DATASETS[$di]} ${MODELS[$mi]} ${PERCENTAGES[$pi]} ${TRIALS[$ti]}" >> "$PARAM_FILE"
        idx=$((idx+1))
      done
    done
  done
done

# MaxArraySize=1001, so we must split into batches of 1000, each with
# its own param file (offset starts from 0 within each batch)
CHUNK=1000
BATCH=0
for START in $(seq 0 $CHUNK $((TOTAL-1))); do
  END=$((START + CHUNK - 1))
  if [ $END -ge $TOTAL ]; then
    END=$((TOTAL - 1))
  fi
  COUNT=$((END - START + 1))

  # Write a batch-specific param file with 0-based indices
  BATCH_PARAM="$RESULTS/param_map_b${BATCH}.txt"
  > "$BATCH_PARAM"
  sed -n "$((START+1)),$((END+1))p" "$PARAM_FILE" | awk -v start="$START" '{printf "%d %s %s %s %s\n", NR-1, $2, $3, $4, $5}' > "$BATCH_PARAM"

  echo "  submitting batch $BATCH: $COUNT jobs (global $START-$END)"
  sbatch --job-name="abl_b${BATCH}" \
    --partition=SallesInfo \
    --array=0-$((COUNT-1)) \
    --ntasks=1 --cpus-per-task=4 --mem=8G \
    --time=01:00:00 \
    --output="$RESULTS/arr_%A_%a.out" \
    --error="$RESULTS/arr_%A_%a.err" \
    "$BASE/run_ablation_array.sh" "$EPOCHS" "$IDENT" "$BATCH"
  BATCH=$((BATCH + 1))
done

echo "Submitted $BATCH array batches. Check: squeue -u ten.nguyen-hanaoka | wc -l"
