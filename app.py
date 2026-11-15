
import streamlit as st
import pandas as pd
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)



#APPLICATION TITLE

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and compare it "
    "with a job description."
)

# EXTRACT TEXT FROM PDF
def extract_resume_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# SKILLS DATABASE

SKILLS_LIST = [
    # Programming Languages
    "Python", "Java", "C++", "C", "C#",
    "JavaScript", "TypeScript", "R", "PHP",
    "Ruby", "Go", "Kotlin", "Swift",

    # Data and Analytics
    "SQL", "MySQL", "PostgreSQL", "MongoDB",
    "Data Analysis", "Data Science",
    "Data Visualization", "Statistics",
    "Power BI", "Tableau", "Excel",
    "NumPy", "Pandas", "Matplotlib",
    "Seaborn", "Power Query",

    # Artificial Intelligence
    "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "NLP",
    "Natural Language Processing",
    "Computer Vision", "TensorFlow",
    "PyTorch", "Scikit-learn",
    "Generative AI", "Large Language Models",

    # Web Development
    "HTML", "CSS", "React", "Angular",
    "Node.js", "Express.js", "Django",
    "Flask", "FastAPI", "Bootstrap",

    # Cloud and DevOps
    "AWS", "Azure", "Google Cloud",
    "Docker", "Kubernetes", "Jenkins",
    "Git", "GitHub", "Linux",
    "CI/CD", "Terraform",

    # Business and Management
    "Communication", "Leadership",
    "Problem Solving", "Teamwork",
    "Project Management", "Critical Thinking",
    "Time Management", "Agile", "Scrum"
]


# EXTRACT SKILLS
def extract_skills(text):

    found_skills = []

    for skill in SKILLS_LIST:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(pattern, text, re.IGNORECASE):

            found_skills.append(skill)

    return found_skills


# RESUME UPLOAD

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)


# JOB DESCRIPTION INPUT

job_description = st.text_area(
    "Paste the Job Description here",
    height=200
)


# ANALYZE RESUME BUTTON

if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:

        st.error("Please upload your resume.")

    elif not job_description.strip():

        st.error("Please enter a job description.")

    else:

        resume_text = extract_resume_text(
            uploaded_file
        )

        if not resume_text.strip():

            st.error(
                "No readable text found. "
                "Try a text-based PDF."
            )

        else:

            # TF-IDF SIMILARITY

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            vectors = vectorizer.fit_transform(
                [resume_text, job_description]
            )

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0]

            score = round(similarity * 100, 2)


            # EXTRACT SKILLS

            resume_skills = extract_skills(
                resume_text
            )

            job_skills = extract_skills(
                job_description
            )


            # MATCHING SKILLS

            matching_skills = [
                skill for skill in job_skills
                if skill in resume_skills
            ]


            # MISSING SKILLS

            missing_skills = [
                skill for skill in job_skills
                if skill not in resume_skills
            ]


            # ADDITIONAL SKILLS

            additional_skills = [
                skill for skill in resume_skills
                if skill not in job_skills
            ]


            # SKILL MATCH SCORE

            if job_skills:

                skill_score = round(
                    len(matching_skills)
                    / len(job_skills) * 100,
                    2
                )

            else:

                skill_score = 0


            # DISPLAY SUCCESS

            st.success("Resume analysis completed!")


            # DISPLAY SCORES

            st.header("📊 Resume Analysis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Resume Match Score",
                f"{score}%"
            )

            col2.metric(
                "Skill Match Score",
                f"{skill_score}%"
            )

            col3.metric(
                "Total Resume Skills",
                len(resume_skills)
            )

            st.progress(skill_score / 100)


            # DISPLAY RESUME SKILLS

            st.header("📋 Skills Found in Resume")

            if resume_skills:

                st.success(
                    f"Found {len(resume_skills)} skills"
                )

                st.write(", ".join(resume_skills))

            else:

                st.warning(
                    "No recognized skills found in resume."
                )

            # DISPLAY JOB SKILLS

            st.header("💼 Skills Required by Job")

            if job_skills:

                st.write(
                    ", ".join(job_skills)
                )

            else:

                st.warning(
                    "No recognized skills found in job description."
                )


            # MATCHING SKILLS

            st.header("✅ Matching Skills")

            if matching_skills:

                st.success(
                    f"{len(matching_skills)} matching skills found"
                )

                st.write(
                    ", ".join(matching_skills)
                )

            else:

                st.warning("No matching skills found.")


            # MISSING SKILLS

            st.header("❌ Missing Skills")

            if missing_skills:

                st.error(
                    f"{len(missing_skills)} skills missing"
                )

                st.write(
                    ", ".join(missing_skills)
                )

            else:

                st.success(
                    "All identified job skills are present!"
                )


            # ADDITIONAL SKILLS

            st.header("⭐ Additional Resume Skills")

            if additional_skills:

                st.info(
                    f"{len(additional_skills)} additional skills found"
                )

                st.write(
                    ", ".join(additional_skills)
                )

            else:

                st.write(
                    "No additional skills found."
                )


            # SKILLS COMPARISON TABLE

            st.header("📑 Skills Comparison")

            comparison_data = []

            for skill in job_skills:

                status = (
                    "Matched"
                    if skill in resume_skills
                    else "Missing"
                )

                comparison_data.append({
                    "Skill": skill,
                    "Status": status
                })

            if comparison_data:

                df = pd.DataFrame(comparison_data)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )


            # DISPLAY EXTRACTED RESUME TEXT

            with st.expander(
                "View Extracted Resume Text"
            ):

                st.text_area(
                    "Resume Content",
                    resume_text,
                    height=250
                )
                
            # INFORMATION MESSAGE

            st.info(
                "This application uses TF-IDF, "
                "cosine similarity and keyword-based "
                "skill extraction. Skill extraction "
                "depends on the predefined skills database. "
                "The scores are not a true measure "
                "of candidate suitability."
            )
