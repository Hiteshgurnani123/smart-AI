import matplotlib.pyplot as plt
import os

def generate_skill_gap_chart(user_skills, predicted_title, job_roles_skills, save_path="static/skill_gap_chart.png"):
    required_skills = set(job_roles_skills.get(predicted_title, []))
    user_skill_set = set([s.strip() for s in user_skills])

    matched = required_skills.intersection(user_skill_set)
    missing = required_skills.difference(user_skill_set)

    labels = ['Matched Skills', 'Missing Skills']
    values = [len(matched), len(missing)]
    colors = ['green', 'red']

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.title(f"Skill Gap for '{predicted_title}'")
    plt.ylabel("Number of Skills")
    plt.tight_layout()

    # Save chart to static folder
    plt.savefig(save_path)
    plt.close()

    return matched, missing
