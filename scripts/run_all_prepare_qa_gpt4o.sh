#!/bin/bash

export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-480}"

start_ts=$(date +%s)

time_periods=("init" "next_week" "next_month" "next_year")

start_persona_id=0
end_persona_id=20  # non-inclusive

for time_period in "${time_periods[@]}"; do
    period_start_ts=$(date +%s)

    # command="python prepare_qa.py --model gpt-4o --action qa \
    #          --api_mode assistants \
    #          --topics bookRecommendation datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
    #                   legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
    #                   studyConsultation therapy travelPlanning writing \
    #          --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --time ${time_period} --clean"

    command="python prepare_qa.py --model gpt-4o --action qa \
             --api_mode assistants \
             --topics legalConsultation financialConsultation medicalConsultation therapy travelPlanning \
             --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --time ${time_period} --clean"

    echo "LLM_TIMEOUT_SEC=${LLM_TIMEOUT_SEC}"
    echo "MODEL=gpt-4o"
    echo "API_MODE=assistants"
    echo "$command"

    eval "$command"

    period_end_ts=$(date +%s)
    period_elapsed=$((period_end_ts - period_start_ts))
    printf "[%s] elapsed: %02d:%02d:%02d\n" "${time_period}" $((period_elapsed/3600)) $(((period_elapsed%3600)/60)) $((period_elapsed%60))
done

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
printf "Total elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
