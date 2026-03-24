#!/bin/bash

export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-480}"
export API_MODE="${API_MODE:-assistants}"

start_ts=$(date +%s)

# Full list of topics for reference
# bookRecommendation datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

# Full list of time_periods for reference
# init next_week next_month next_year

# Define the list of time periods
time_periods=("init" "next_week" "next_month" "next_year")

start_persona_id=0
end_persona_id=20  # non-inclusive

# Loop over each time period
for time_period in "${time_periods[@]}"; do
    period_start_ts=$(date +%s)

    # Construct the command
    command="python prepare_qa.py --model gpt-4o --action qa \
             --api_mode ${API_MODE} \
             --topics bookRecommendation datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
                      legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
                      studyConsultation therapy travelPlanning writing \
             --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --time ${time_period} --clean"

    # Print the command for debugging/logging purposes
    echo "LLM_TIMEOUT_SEC=${LLM_TIMEOUT_SEC}"
    echo "API_MODE=${API_MODE}"
    echo "$command"

    # Execute the command
    eval "$command"

    period_end_ts=$(date +%s)
    period_elapsed=$((period_end_ts - period_start_ts))
    printf "[%s] elapsed: %02d:%02d:%02d\n" "${time_period}" $((period_elapsed/3600)) $(((period_elapsed%3600)/60)) $((period_elapsed%60))
done

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
printf "Total elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
