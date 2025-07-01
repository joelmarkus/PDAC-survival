#!/bin/bash
## Script to generate a combined list of elastic-net and DEGs
cd /fh/scratch/delete90/gujral_t/joel/PDAC_survival/Joel
## example: sbatch -c 5 -t 7-0 run_SVC.sh
echo 'Input Required (none for now)'

ml Anaconda3
source activate opt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# Read the file and loop over the first 20 rows
input_file="Misc/successfully_generated_combined_lists.csv"
count=0
start_row=4
end_row=7


while IFS= read -r drug_name; do
    if [ $count -ge $((start_row - 1)) ] && [ $count -lt $end_row ]; then
        drug_name=$(echo $drug_name | tr -d '\r')
        echo $drug_name
        echo "Processing drug: $drug_name"

        # Execute the Python scripts
        # python SVC_RFE.py "$drug_name"
        # python SVC_RFE_finalrun.py "$drug_name"
        python SVC_RFE_finalrun_dummyranks.py "$drug_name"
        python SVC_Unseen.py "$drug_name"
    fi
    ((count++))
done < <(tail -n +1 "$input_file")  # Read all lines including the first one

conda deactivate

echo "Deactivated the envt. Done with all the drugs from row $start_row to row $end_row"
