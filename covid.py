import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def bayes_theorem(prior, sensitivity, specificity):
    false_positive_rate = 1 - specificity
    p_B = (sensitivity * prior) + (false_positive_rate * (1 - prior))
    posterior = (sensitivity * prior) / p_B
    return posterior

st.title("COVID-19 Probability Calculator")

st.sidebar.header("Input Parameters")
prior = st.sidebar.slider("Prevalence P(A)", 0.0, 1.0, 0.04, 0.01)
sensitivity = st.sidebar.slider("Sensitivity P(B|A)", 0.0, 1.0, 0.73, 0.01)
specificity = st.sidebar.slider("Specificity P(¬B|¬A)", 0.0, 1.0, 0.95, 0.01)
st.sidebar.text("María Camila Villamizar Villamizar")


posterior = bayes_theorem(prior, sensitivity, specificity)

st.subheader("Results")
st.write(f"Given a positive test result, the probability of actually having COVID-19 is: **{posterior:.4f}**")


# Plot sensitivity vs posterior probability
sens_values = np.linspace(0, 1, 100)
posteriors = [bayes_theorem(prior, s, specificity) for s in sens_values]

fig, ax = plt.subplots()
ax.plot(sens_values, posteriors, label='Posterior Probability', color='purple')
ax.set_xlabel("Sensitivity")
ax.set_ylabel("P(A|B)")
ax.set_title("Effect of Sensitivity on Posterior Probability")
ax.legend()
st.pyplot(fig)

# More comprehensive plot: Sensitivity and Specificity vs Posterior Probability
spec_values = np.linspace(0, 1, 100)
posterior_matrix = np.array([[bayes_theorem(prior, s, sp) for sp in spec_values] for s in sens_values])

fig2, ax2 = plt.subplots()
c = ax2.contourf(spec_values, sens_values, posterior_matrix, cmap="viridis")
fig2.colorbar(c, label="Posterior Probability")
ax2.set_xlabel("Specificity")
ax2.set_ylabel("Sensitivity")
ax2.set_title("Effect of Sensitivity and Specificity on Posterior Probability")
st.pyplot(fig2)

# Plot True Positives, False Positives, True Negatives, and False Negatives
true_positives = sensitivity * prior
false_negatives = (1 - sensitivity) * prior
false_positives = (1 - specificity) * (1 - prior)
true_negatives = specificity * (1 - prior)

labels = ['True Positives', 'False Negatives', 'False Positives', 'True Negatives']
values = [true_positives, false_negatives, false_positives, true_negatives]

fig3, ax3 = plt.subplots()
cmap = plt.get_cmap("viridis")
colors = [cmap(0.8), cmap(0.6), cmap(0.4), cmap(0.2)]
ax3.bar(labels, values, color=colors)
ax3.set_ylabel("Proportion")
ax3.set_title("Breakdown of Test Outcomes")
st.pyplot(fig3)
