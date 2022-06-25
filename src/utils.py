MODEL_CODES = ("KNN", "DT", "CLB", "PLT", "MIT", "F", "G", "D", "GR", "FH", "TA", "R")

MODEL_SHORT_NAMES = (
    "KNN",
    "DT",
    "BART",
    "PLT",
    "BGP",
    "FAL",
    "FALGP",
    "FALDN",
    "GRYF",
    "FALH",
    "GPTAF",
    "RND",
)

MODEL_FULL_NAMES = (
    "K-Nearest Neighbors",
    "Decision Trees",
    "Bayesian Additive Regression Trees with Transfer Learning",
    "Atinary Falcon",
    "Atinary Falcon GPBO",
    "Atinary Falcon DNGO",
    "Gryffin",
    "Atinary Falcon leveraging historical data",
    "Gaussian Processes with Transfer Acquisition Functions",
    "Random",
)

draw0_colors = [
    "rgba(60, 180, 75,1.0)",
    "rgba(255, 225, 25,1.0)",
    "rgba(0, 130, 200,1.0)",
    "rgba(245, 130, 48,1.0)",
    "rgba(145, 30, 180,1.0)",
    "rgba(70, 240, 240,1.0)",
    "rgba(240, 50, 230,1.0)",
    "rgba(210, 245, 60,1.0)",
    "rgba(250, 190, 212,1.0)",
    "rgba(0, 128, 128,1.0)",
    "rgba(220, 190, 255,1.0)",
    "rgba(170, 110, 40,1.0)",
    "rgba(255, 250, 200,1.0)",
    "rgba(128, 0, 0,1.0)",
    "rgba(170, 255, 195,1.0)",
    "rgba(128, 128, 0,1.0)",
    "rgba(255, 215, 180,1.0)",
    "rgba(0, 0, 128,1.0)",
    "rgba(128, 128, 128,1.0)",
    "rgba(255, 255, 255,1.0)",
    "rgba(0, 0, 0,1.0)",
    "rgba(230, 25, 75,1.0)",
]

draw1_colors = [
    "rgba(60, 180, 75, 0.75)",
    "rgba(255, 225, 25, 0.75)",
    "rgba(0, 130, 200, 0.75)",
    "rgba(245, 130, 48, 0.75)",
    "rgba(145, 30, 180, 0.75)",
    "rgba(70, 240, 240, 0.75)",
    "rgba(240, 50, 230, 0.75)",
    "rgba(210, 245, 60, 0.75)",
    "rgba(250, 190, 212, 0.75)",
    "rgba(0, 128, 128, 0.75)",
    "rgba(220, 190, 255, 0.75)",
    "rgba(170, 110, 40, 0.75)",
    "rgba(255, 250, 200, 0.75)",
    "rgba(128, 0, 0, 0.75)",
    "rgba(170, 255, 195, 0.75)",
    "rgba(128, 128, 0, 0.75)",
    "rgba(255, 215, 180, 0.75)",
    "rgba(0, 0, 128, 0.75)",
    "rgba(128, 128, 128, 0.75)",
    "rgba(255, 255, 255, 0.75)",
    "rgba(0, 0, 0, 0.75)",
    "rgba(230, 25, 75, 0.75)",
]

COLORS = dict(zip([m + "0" for m in MODEL_CODES], draw0_colors[: len(MODEL_CODES)]))
COLORS1 = dict(zip([m + "1" for m in MODEL_CODES], draw1_colors[: len(MODEL_CODES)]))
COLORS2 = dict(zip([m for m in MODEL_CODES], draw0_colors[: len(MODEL_CODES)]))
COLORS.update(COLORS1)
COLORS.update(COLORS2)

AMINE_INCHIS = ['ZKRCWINLLKOVCL-UHFFFAOYSA-N','JMXLWMIFDJCGBV-UHFFFAOYSA-N', 'NJQKYAASWUGXIT-UHFFFAOYSA-N','HJFYRMFYQMIZDG-UHFFFAOYSA-N']
AMINE_NAMES = ['4-Chlorophenethylammonium iodide', 'Dimethylammonium iodide', '4-Chlorophenylammonium iodide', '4-Hydroxyphenethylammonium iodide']

AMINES = dict(zip(AMINE_NAMES, AMINE_INCHIS))