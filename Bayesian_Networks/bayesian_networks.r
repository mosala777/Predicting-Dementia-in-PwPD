library(bnlearn)
library(dplyr)
library(tidyverse)
library(readr)

#' Load and preprocess data
#'
#' @param file_path Path to the CSV file.
#' @param exclude_columns Columns to exclude from the dataset.
#' @return A data frame with the selected columns.
load_and_preprocess_data <- function(file_path, exclude_columns = c("sample")) {
  data <- read_csv(file_path)
  data <- select(data, -all_of(exclude_columns))
  return(data)
}

#' Select top features
#'
#' @param data A data frame containing the dataset.
#' @param feature_list A vector of column names to select as features.
#' @return A data frame with the selected top features.
select_top_features <- function(data, feature_list) {
  top_features <- select(data, all_of(feature_list))
  return(top_features)
}

#' Create a blacklist for Bayesian Network structure learning
#'
#' @param features A data frame of features.
#' @param targets A vector of target variable names to blacklist connections.
#' @return A data frame representing the blacklist.
create_blacklist <- function(features, targets) {
  blacklist <- data.frame()
  for (target in targets) {
    blacklist <- rbind.data.frame(blacklist, data.frame(from = colnames(features), to = target))
  }
  return(blacklist)
}

#' Perform bootstrap strength calculations
#'
#' @param features A data frame of features.
#' @param blacklist A data frame representing the blacklist.
#' @param algorithm Algorithm to use for structure learning (default: "tabu").
#' @param R Number of bootstrap replications.
#' @param maxp Maximum number of parents.
#' @return A data frame of bootstrap strength results.
perform_bootstrap <- function(features, blacklist, algorithm = "tabu", R = 1000, maxp = 5) {
  bootstrap_strength <- boot.strength(
    features,
    algorithm = algorithm,
    R = R,
    algorithm.args = list(maxp = maxp, blacklist = blacklist)
  )
  return(bootstrap_strength)
}

#' Average networks and filter edges
#'
#' @param bootstrap_strength A data frame of bootstrap strength results.
#' @param threshold Strength threshold for averaging networks.
#' @return A list containing the averaged network and bootstrapped edges.
average_network <- function(bootstrap_strength, threshold = 0.5) {
  averaged_bn <- averaged.network(bootstrap_strength, threshold = threshold)
  bootstrapped_edges <- bootstrap_strength[bootstrap_strength$strength > threshold & bootstrap_strength$direction >= 0.5, ]
  return(list(averaged_bn = averaged_bn, bootstrapped_edges = bootstrapped_edges))
}

#' Fit the Bayesian Network
#'
#' @param network A Bayesian network object.
#' @param features A data frame of features.
#' @return A fitted Bayesian network object.
fit_bayesian_network <- function(network, features) {
  fitted_bn <- bn.fit(network, features)
  return(fitted_bn)
}

#' Run Bayesian Network Analysis
#'
#' @param file_path Path to the CSV file.
#' @param feature_list A vector of column names to use as features.
#' @param targets A vector of target variable names for blacklist connections.
#' @param threshold Strength threshold for averaging networks.
#' @param R Number of bootstrap replications.
#' @param maxp Maximum number of parents.
#' @return A list containing the fitted Bayesian network and bootstrapped edges.
run_bayesian_network_analysis <- function(file_path, feature_list, targets, threshold = 0.5, R = 1000, maxp = 5) {
  # Load and preprocess data
  data <- load_and_preprocess_data(file_path)

  # Select top features
  top_features <- select_top_features(data, feature_list)

  # Create blacklist
  blacklist <- create_blacklist(top_features, targets)

  # Perform bootstrapping
  bootstrap_strength <- perform_bootstrap(top_features, blacklist, R = R, maxp = maxp)

  # Average networks and filter edges
  averaged_result <- average_network(bootstrap_strength, threshold)
  
  # Fit Bayesian Network
  fitted_bn <- fit_bayesian_network(averaged_result$averaged_bn, top_features)

  return(list(fitted_bn = fitted_bn, edges = averaged_result$bootstrapped_edges))
}

# Example usage
file_path <- "<path_to_csv_file>"
feature_list <- c("Feature1", "Feature2", "Feature3")
targets <- c("Target1", "Target2")

result <- run_bayesian_network_analysis(
  file_path = file_path,
  feature_list = feature_list,
  targets = targets,
  threshold = 0.5,
  R = 1000,
  maxp = 5
)
