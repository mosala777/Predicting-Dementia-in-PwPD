import genal
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import argparse

def initialize_genal_plink(plink_path):
    """Initialize Genal with the PLINK path."""
    genal.set_plink(path=plink_path)

def load_gwas_data(file_path):
    """Load GWAS data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_gwas_data(gwas):
    """Preprocess GWAS data using Genal."""
    return genal.Geno(
        gwas,
        CHR="chr",
        POS="position",
        EA="ea",
        NEA="nea",
        BETA="beta",
        SE="se",
        P="p",
        EAF="eaf",
        keep_columns=False
    ).preprocess_data(preprocessing="Fill_delete")

def load_and_encode_phenotypes(file_path, encode_covariates):
    """Load phenotype data and encode specified categorical covariates."""
    df_pheno = pd.read_csv(file_path)
    le = LabelEncoder()
    for covariate in encode_covariates:
        if covariate in df_pheno.columns:
            df_pheno[covariate] = le.fit_transform(df_pheno[covariate])
    return df_pheno

def perform_association_tests(geno, df_pheno, phenotype, association_covariates, genetic_files_path):
    """Perform association tests using Genal."""
    geno.set_phenotype(df_pheno, PHENO=phenotype, IID="FID")
    geno.association_test(covar=association_covariates, path=genetic_files_path)

def clump_and_query_outcome(pheno1, pheno2):
    """Perform clumping and query outcome data."""
    clumped = pheno1.clump(p1=5e-8, r2=0.1, kb=250, reference_panel="eur")
    clumped.query_outcome(pheno2, proxy=True, reference_panel="eur", kb=5000, r2=0.6, window_snps=5000)
    return clumped

def perform_mr(clumped, exposure_name, outcome_name):
    """Perform Mendelian Randomization using Genal."""
    return clumped.MR(action=2, exposure_name=exposure_name, outcome_name=outcome_name)

def save_results(data, file_path):
    """Save results to a CSV file."""
    data.to_csv(file_path, index=False)

def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, "w") as f:
        json.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Run Mendelian Randomization analysis using Genal.")
    parser.add_argument("plink_path", type=str, help="Path to the PLINK executable.")
    parser.add_argument("genetic_files_path", type=str, help="Path to the genetic data files.")
    parser.add_argument("gwas_path", type=str, help="Path to the GWAS summary statistics CSV file.")
    parser.add_argument("phenotypes_path", type=str, help="Path to the phenotype data CSV file.")
    parser.add_argument("exposure", type=str, help="Column name for the exposure in phenotype data.")
    parser.add_argument("outcome", type=str, help="Column name for the outcome in phenotype data.")
    
    parser.add_argument("--encode_covariates", type=str, nargs='*', default=[], help="List of covariates to be label-encoded.")
    parser.add_argument("--association_covariates", type=str, nargs='*', default=[], help="List of covariates to be used in association tests.")
    args = parser.parse_args()

    # Initialize Genal
    initialize_genal_plink(args.plink_path)

    # Load and preprocess GWAS data
    gwas = load_gwas_data(args.gwas_path)
    geno = preprocess_gwas_data(gwas)

    # Load and encode phenotype data
    df_pheno = load_and_encode_phenotypes(args.phenotypes_path, args.encode_covariates)

    # Copy geno objects for each phenotype
    pheno1 = geno.copy()
    pheno2 = geno.copy()

    # Perform association tests
    perform_association_tests(pheno1, df_pheno, args.exposure, args.association_covariates, args.genetic_files_path)
    perform_association_tests(pheno2, df_pheno, args.outcome, args.association_covariates, args.genetic_files_path)

    # Perform clumping and query outcomes
    clumped = clump_and_query_outcome(pheno1, pheno2)

    # Perform MR analysis
    mr_results = perform_mr(clumped, args.exposure, args.outcome)

    # Save MR results
    save_results(clumped.MR_results[1], "snp_effects.csv")
    save_results(clumped.MR_data[1], "summary_snps.csv")

    # Perform MR-PRESSO
    mr_presso = clumped.MRpresso(action=2, n_iterations=30000)
    save_results(mr_presso[0], "mr_presso_results.csv")
    save_results(mr_presso[2], "mr_presso_snps.csv")
    save_json(mr_presso[3], "mr_presso_distortion.json")

if __name__ == "__main__":
    main()
