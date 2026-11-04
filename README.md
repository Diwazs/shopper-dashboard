# Mathematical Foundation of Business Analytics (ANT3207)
## Online Shoppers Purchasing Intention — Interactive Analytics Dashboard

**Course:** Mathematical Foundation of Business Analytics (ANT3207)  
**Program:** BBA (Hons) Analytics · Semester V  
**Institution:** T. A. Pai Management Institute (TAPMI), Bengaluru  
**Faculty:** Dr. Bhupesh Dutt Sharma  
**Group:** WAS Group-9  

---

## 📌 Executive Summary
This interactive Streamlit dashboard presents an end-to-end mathematical and business analysis of 12,330 e-commerce sessions from the UCI *Online Shoppers Purchasing Intention* dataset. It connects real business acquisition decisions (traffic channel ROI, user segmentation, browsing duration) with the mathematical concepts taught across all 30 sessions of the MFBA curriculum.

---

## 🏗️ Course Module Mapping

| Module & Sessions | Mathematical Concept | Implementation in Dashboard & Script |
| :--- | :--- | :--- |
| **Module I: Probability Foundations**<br/>*(Sessions 1–7)* | • Maximum Likelihood Estimation (MLE)<br/>• Wilson Score Intervals<br/>• Hoeffding Concentration Inequality (PAC)<br/>• Empirical Bayes Beta-Binomial Shrinkage<br/>• Chi-Square Test of Independence | • Channel conversion scorecard comparing Wald, Wilson, and Hoeffding bounds.<br/>• Nelder-Mead MLE fitting of prior hyperparameters ($\hat{\alpha} = 3.473, \hat{\beta} = 18.743$).<br/>• Monte Carlo posterior draws proving $P(\text{Type 8} > \text{Type 13}) = 1.000$. |
| **Module II: Linear Algebra**<br/>*(Sessions 8–15)* | • Feature Standardization ($Z$-scores)<br/>• SVD by Hand ($Z = U \Sigma V^T$)<br/>• Covariance Eigendecomposition<br/>• PCA Factor Loadings & Scree Plots<br/>• Eckart-Young Rank-$k$ Approximation | • Hand SVD matching `scikit-learn` to $\pm 1.14 \times 10^{-15}$.<br/>• 3 components capture 88.7% of total variance (Kaiser criterion $\lambda > 1$).<br/>• Loadings heatmap across 8 browsing metrics.<br/>• Frobenius norm rank-$k$ relative error curve. |
| **Module IV: Clustering & Segments**<br/>*(Sessions 24–30)* | • $k$-Means (Voronoi Partitioning)<br/>• Gaussian Mixture Models (EM Algorithm)<br/>• Silhouette Coefficient & BIC<br/>• Adjusted Rand Index (ARI) | • Optimal $K=4$ clustering (Silhouette peak = 0.526).<br/>• ARI = 0.776 confirming spherical vs elliptical geometry alignment.<br/>• 4 Personas: *Bouncers*, *Light Browsers*, *Account-Engaged*, *Deep Researchers* (34.2% conversion, $2.21\times$ lift). |
| **Module III: Regression & Optimization**<br/>*(Sessions 16–23)* | • Polynomial Design Matrix Expansion<br/>• 10-Fold Cross-Validation (1-SE Rule)<br/>• Closed-Form OLS with $t$-statistics<br/>• Gradient Descent from Scratch<br/>• Lipschitz Continuous Step Size $\eta = 1/\lambda_{\max}(H)$<br/>• Minibatch SGD with Step Decay | • 10-fold CV selects Degree 6 under 1-SE rule ($R^2 = 82.6\%$).<br/>• Theoretical step size $\eta = 1/\lambda_{\max}(H) \approx 0.3085$ ($\lambda_{\max} = 3.2412$).<br/>• Practical convergence ($< 10^{-6}$ loss gap) achieved at iteration 2,272; full 60,000 iterations match closed-form OLS to $3.89 \times 10^{-14}$.<br/>• Minibatch SGD (batch size 64) converges to neighborhood within 30 epochs. |

---

## 🚀 How to Run the Dashboard

1. **Activate the Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Verify Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Analytical Pipeline (Optional / Verification):**
   ```bash
   python analysis.py online_shoppers_intention.csv 4 tables_and_figures
   ```

4. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
shopper-dashboard/
├── online_shoppers_intention.csv       # UCI benchmark dataset (12,330 sessions)
├── analysis.py                         # Reference 432-line mathematical engine
├── app.py                              # Upgraded interactive Streamlit dashboard
├── README.md                           # Project documentation
├── requirements.txt                    # Python environment specifications
├── results.json                        # Precomputed analytical outputs (56 metric keys)
├── tables_and_figures/                 # 7 PNG figures and 8 verified CSV tables
│   ├── fig1_channel_scorecard.png
│   ├── fig2_pca_scree.png
│   ├── fig3_pca_loadings.png
│   ├── fig4_choose_k.png
│   ├── fig5_segments.png
│   ├── fig6_cv_degree.png
│   ├── fig7_gd_convergence.png
│   ├── channel_scorecard_all.csv
│   ├── pca_loadings.csv
│   ├── behaviour_correlations.csv
│   ├── segment_profile.csv
│   ├── channel_segment_mix.csv
│   ├── conv_by_segment_channel.csv
│   ├── regression_coefficients.csv
│   └── segment_labels.csv
├── docs/                               # Academic guides, proposals & course slides
│   ├── MFBA_Project_Report_WAS_Group9.docx
│   ├── MFBA_Worked_Calculations_Viva_Guide.docx
│   ├── main_proposal_for_maths.pdf
│   └── ANT3207_Lecture_Slides.pdf
├── reference/                          # Backups and baseline scripts
│   ├── app_backup.py                   # Original app.py backup
│   ├── analysis_original_432.py
│   └── analysis_lightweight.py
└── venv/                               # Python 3.13.5 virtual environment
```

---

## 🎓 Academic Viva Reference Highlights
* **Why $\ln(1+x)$?** Compresses heavy right skew (skewness $+4.2 \to +0.8$) while handling zeros gracefully ($\ln 1 = 0$).
* **Why Empirical Bayes?** Fixes small-sample noise in MLE ($n < 50$), pulling rates toward the global prior mean ($15.63\%$) via Beta-Binomial marginal log-likelihood optimization.
* **Why $K=4$?** Elbow inflection point and silhouette coefficient peak ($0.526$). While GMM BIC continues decreasing up to $K=8$, additional Gaussians merely model non-Gaussian skewness rather than managerially actionable customer types.
* **Why Degree 6 over Degree 8?** While Degree 8 yields the absolute minimum CV RMSE, the canonical 1-Standard-Error rule selects Degree 6, avoiding boundary Runge oscillations.
* **Why $\eta = 1/\lambda_{\max}(H)$?** Quadratic loss has Hessian $H = \frac{1}{n} M_s^T M_s$. The Lipschitz constant of the gradient is $L = \lambda_{\max}(H) \approx 3.2412$, yielding the theoretical safe step size $\eta = 1/3.2412 \approx 0.3085$ (or 0.309 as rounded in the viva guide), which mathematically guarantees monotonic descent without divergence. Practical convergence ($< 10^{-6}$) is achieved in 2,272 iterations, while 60,000 iterations reach $3.89 \times 10^{-14}$ agreement with closed-form OLS.
