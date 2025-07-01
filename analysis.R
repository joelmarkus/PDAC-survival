# Load required libraries
library(ggvenn)
library(pheatmap)
library(ggplot2)
library(grid)
library(gridExtra)
library(grid)

# Read the CSV file
data <- read.csv("marker_genes_list.csv", check.names = F)

# Create a list of sets
sets <- list(
  `three-year` = unique(data$`three-year`),
  `five-year` = unique(data$`five-year`),
  `Yuqi-genes` = unique(data$Yuqi)
)

sets <- lapply(sets, function(x) x[x != ""])

# Plot the Venn diagram
venn_plot <- ggvenn(sets, text_size = 7)

ggsave("venn_diagram_new.png", venn_plot, width = 8, height = 7, bg = "white")

#run correlation of sorts-----
##correlation with Yuqi's genes----
dat <- read.csv("paad_logTPM_treatment_patient_protcode_jv.csv", check.names = F, row.names = 1)
gene_list <- read.csv("marker_genes_list.csv")
survival <- read.csv("survival.csv", check.names = F, row.names = 1)

dat_jv <- dat[rownames(dat) %in% gene_list$five.year,]
dat_yk <- dat[rownames(dat) %in% gene_list$Yuqi,]

correlation_matrix <- cor(t(dat_jv), t(dat_yk), method = "pearson")


heatmap1 <- pheatmap(correlation_matrix, 
                     cluster_rows = TRUE, 
                     cluster_cols = TRUE,
                     display_numbers = FALSE, 
                     main = "Pearson correlation",
                     margins = c(100, 100))


# Save as PNG
png("R_cor_genes.png", width = 800, height = 700)
grid.draw(heatmap1$gtable)
grid.text("Joel RFE 5-year genes", x = 0.5, y = 0, gp = gpar(fontsize = 15), just = "center")
grid.text("Yuqi 60 5-year genes", x = 0, y = 0.5, gp = gpar(fontsize = 15), rot = 90, just = "center")
dev.off()

# Save as SVG
cairo_ps("R_cor_genes.eps", width = 8, height = 7)
grid.draw(heatmap1$gtable)
grid.text("Joel RFE 5-year genes", x = 0.5, y = 0, gp = gpar(fontsize = 15), just = "center")
grid.text("Yuqi 60 5-year genes", x = 0, y = 0.5, gp = gpar(fontsize = 15), rot = 90, just = "center")
dev.off()

##correlation with survival----
survival_status <- as.numeric(survival[1, ])

# Calculate Spearman correlation for each gene (row) in dat_jv with survival status
correlation_results <- apply(dat_jv, 1, function(gene) {
  cor(gene, survival_status, method = "pearson", use = "complete.obs")
})


# Optional: create a data frame to view results neatly
correlation_matrix <- as.matrix(correlation_results, dimnames = list("Correlation", rownames(dat_jv)))

# Plot the heatmap using pheatmap
heatmap2 <- pheatmap(correlation_matrix, 
         cluster_rows = TRUE, 
         cluster_cols = FALSE, 
         #display_numbers = TRUE, 
         main = "Genes - Survival Cor")

# Save as PNG
png("R_cor_genes_survival.png", width = 175, height = 700)
grid.draw(heatmap2$gtable)
dev.off()

# Save as SVG
cairo_ps("R_cor_genes_survival.eps", width = 2.5, height = 7)
grid.draw(heatmap2$gtable)
dev.off()
