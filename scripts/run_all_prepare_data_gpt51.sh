#!/bin/bash

export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-480}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

start_ts=$(date +%s)

start_persona_id=0
end_persona_id=10  # non-inclusive

skip_existing_flag=""
if [ "${SKIP_EXISTING}" = "1" ]; then
  skip_existing_flag="--skip_existing"
fi

# command="python prepare_data.py --model gpt-5-mini \
#          --api_mode responses \
#          --topics bookRecommendation datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
#                   legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
#                   studyConsultation therapy travelPlanning writing \
#          --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --output_dir data/output/ ${skip_existing_flag}"


command="python prepare_data.py --model gpt-5-mini \
         --api_mode responses \
         --topics legalConsultation financialConsultation medicalConsultation therapy travelPlanning \
         --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --workers 10 --output_dir data/output/ ${skip_existing_flag}"

echo "LLM_TIMEOUT_SEC=${LLM_TIMEOUT_SEC}"
echo "MODEL=gpt-5-mini"
echo "API_MODE=responses"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "$command"

eval "$command"

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
printf "Total elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
