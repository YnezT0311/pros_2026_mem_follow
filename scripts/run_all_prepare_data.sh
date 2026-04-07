#!/bin/bash

export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-480}"
# export API_MODE="${API_MODE:-auto}"
export API_MODE="${API_MODE:-responses}"
export TMP_DIR="${TMP_DIR:-tmp}"
export TOPICS="${TOPICS:-legalConsultation financialConsultation medicalConsultation therapy travelPlanning}"
export WORKERS="${WORKERS:-10}"

mkdir -p "${TMP_DIR}"

start_ts=$(date +%s)

# Full list of topics for reference
# bookRecommendation datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

start_persona_id=0
end_persona_id=10  # non-inclusive

# Construct the command
# Print the command for debugging/logging purposes
echo "LLM_TIMEOUT_SEC=${LLM_TIMEOUT_SEC}"
echo "API_MODE=${API_MODE}"
echo "TOPICS=${TOPICS}"
echo "WORKERS=${WORKERS}"

for topic in ${TOPICS}; do
  command="python prepare_data.py --model gpt-5-mini \
           --api_mode ${API_MODE} \
           --topics ${topic} \
           --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 \
           --workers ${WORKERS} --output_dir data/output/ --skip_existing"
  echo "$command"
  eval "$command"
done

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
printf "Total elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
