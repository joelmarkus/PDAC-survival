library(data.table)
library(dplyr)

#prepare expression file:
df <- read.csv("paad_logTPM_treatment_patient_protcode_jv.csv", row.names = 1, check.names = F)
# Function to calculate the standard deviation of each row
row_sd <- function(x) {
  apply(x, 1, sd)
}

# Add a new column for the row standard deviations
df <- df %>%
  mutate(row_sd = row_sd(.))

# Filter the rows where the standard deviation is greater than or equal to 1
filtered_df <- df %>%
  filter(row_sd >= 1) %>%
  select(-row_sd)  # Remove the temporary row_sd column

write.csv(rownames(filtered_df), "five_year_survival/combined_list1.csv", row.names = F)

#Create survival information:
survival <- read.csv("../paad_survival.csv")

# Assuming your data frame is named df
survival <- survival %>%
  mutate(
    five_year_survival = case_when(
      is.na(dss_status) | is.na(month) ~ NA_real_,
      dss_status == 0 ~ 0,
      dss_status == 1 & month >= 60 ~ 0,
      dss_status == 1 & month < 60 ~ 1,
      TRUE ~ NA_real_
    ),
    three_year_survival = case_when(
      is.na(dss_status) | is.na(month) ~ NA_real_,
      dss_status == 0 ~ 0,
      dss_status == 1 & month >= 36 ~ 0,
      dss_status == 1 & month < 36 ~ 1,
      TRUE ~ NA_real_
    ),
    one_year_survival = case_when(
      is.na(dss_status) | is.na(month) ~ NA_real_,
      dss_status == 0 ~ 0,
      dss_status == 1 & month >= 12 ~ 0,
      dss_status == 1 & month < 12 ~ 1,
      TRUE ~ NA_real_
    )
  )

fwrite(survival, "../paad_survival_all.csv")



#rearrange paad_log_exp values to match survival
df <- read.csv("paad_logTPM_treatment_patient_protcode_jv.csv", row.names = 1, check.names = F)
survival_order <- survival$X
df <- df %>% select(all_of(survival_order))
write.csv(df, "paad_logTPM_treatment_patient_protcode_jv.csv", row.names = T) #, check.names = F
