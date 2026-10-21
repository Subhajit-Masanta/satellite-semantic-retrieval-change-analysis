"""Build the Project-I style report in the BCSE497J template format,
but for subject code BCSE407L (Computer Vision), in simple student English."""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from pathlib import Path

ROOT = Path(r"C:\Users\subha\Desktop\projects\Computer Vision")
FIGS = ROOT / "report_figs"
OUT = ROOT / "BCSE407L_Project_Report_Satellite_Semantic_Retrieval_v2.docx"
RFIG = ROOT / "results" / "figs"
import json, sys
sys.path.insert(0, str(ROOT / "report_tools"))
PG_FILE = ROOT / "report_tools" / "pages.json"
PAGES = json.loads(PG_FILE.read_text()) if PG_FILE.exists() else {}

doc = Document()

# ------------------------------------------------------------------ base style
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
st.paragraph_format.space_after = Pt(6)

for s in doc.sections:
    s.top_margin = Inches(1); s.bottom_margin = Inches(1)
    s.left_margin = Inches(1.25); s.right_margin = Inches(1)


def P(text="", size=12, bold=False, italic=False, align="left", spacing=1.15,
      space_after=6, upper=False, color=None):
    p = doc.add_paragraph()
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "just": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    p.paragraph_format.line_spacing = spacing
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text.upper() if upper else text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color:
        r.font.color.rgb = color
    return p


def H1(text):            # 14, bold, upper, 1.5
    return P(text, size=14, bold=True, upper=True, spacing=1.5, space_after=8)


def H2(text):            # 13, bold, title case, 1.5
    return P(text, size=13, bold=True, spacing=1.5, space_after=6)


def H3(text):            # 12, bold italic, 1.5
    return P(text, size=12, bold=True, italic=True, spacing=1.5, space_after=4)


def body(text):
    return P(text, size=12, align="just", spacing=1.15)


def bullet(text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(3)
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
        r.font.name = "Times New Roman"; r.font.size = Pt(12)
    r = p.add_run(text)
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
    return p


def numbered(text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
    return p


def figure(path, caption, width=6.1):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(caption, size=11, align="center", spacing=1.0, space_after=10)


def table(headers, rows, widths=None, font=10.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        pr = c.paragraphs[0]; pr.paragraph_format.space_after = Pt(2)
        r = pr.add_run(h); r.bold = True
        r.font.name = "Times New Roman"; r.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            pr = cells[i].paragraphs[0]; pr.paragraph_format.space_after = Pt(2)
            r = pr.add_run(str(v))
            r.font.name = "Times New Roman"; r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return t


# ============================================================== TITLE PAGE
P("BCSE407L – Computer Vision", size=14, bold=True, align="center", spacing=1.5)
P("Project Report", size=13, bold=True, align="center", spacing=1.5, space_after=10)

P("SEMANTIC RETRIEVAL AND MULTI-TEMPORAL CHANGE ANALYSIS OF SATELLITE IMAGERY "
  "USING A SHARED FOUNDATION-MODEL EMBEDDING SPACE",
  size=16, bold=True, align="center", spacing=1.5, space_after=12)

P("Submitted by", size=12, align="center", spacing=1.5, space_after=8)
tbl = table(["Reg. No.", "Student Name"],
            [["23BCE0257", "AKANKSHA BHARTI"],
             ["23BCE0292", "SUBHAJIT MASANTA"],
             ["23BCE0318", "GONTLA VENKATA SIVA SAI NIKHIL"],
             ["23BCE2027", "ARJUN PRATAP SINGH"]],
            widths=[1.6, 3.4], font=12)
tbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
P(space_after=4)

P("Under the Supervision of", size=12, align="center", spacing=1.5, space_after=6)
P("Dr. <Supervisor Name>", size=12, bold=True, align="center", spacing=1.5, space_after=0)
P("<Designation>", size=12, align="center", spacing=1.5, space_after=0)
P("School of Computer Science and Engineering", size=12, align="center",
  spacing=1.5, space_after=8)

P("In partial fulfilment of the requirements for the degree of", size=12,
  align="center", spacing=1.5, space_after=6)
P("B.Tech.", size=13, bold=True, align="center", spacing=1.5, space_after=0)
P("in", size=12, align="center", spacing=1.5, space_after=0)
P("Computer Science and Engineering", size=13, bold=True, align="center",
  spacing=1.5, space_after=8)
P("School of Computer Science and Engineering", size=12, align="center",
  spacing=1.5, space_after=12)
P("October 2026", size=12, bold=True, align="center", spacing=1.5)

doc.add_page_break()

# ================================================================= ABSTRACT
H1("Abstract")
body("Satellites photograph the Earth every day, so the amount of stored "
     "satellite imagery is now very large. These archives can only be searched "
     "by date, coordinates or satellite name. They cannot be searched by what is "
     "inside the picture, so a person must already know where to look. Finding "
     "what changed in a place is even harder, because someone has to compare two "
     "images by eye.")
body("In this project we build a system that tries to solve both problems "
     "together. We cut every image into small tiles and pass each tile through a "
     "pre-trained CLIP model, which turns the tile into a list of numbers called "
     "an embedding. We store all embeddings once in one shared index and use it "
     "for two jobs. First, a user can type words such as \"a river\" or give an "
     "example tile, and the system returns matching tiles. Second, for the same "
     "place at two dates, we compare the two embeddings to see if something "
     "changed. We also remove false alarms caused by brightness differences, "
     "cloud and shadow. We do not train any model.")
body("We tested the system on the OSCD dataset, which has 24 pairs of Sentinel-2 "
     "city images with correct change masks. Our results are mixed and we report "
     "them as they are. Searching for the same place across two dates worked "
     "well: the correct tile came first 46% of the time out of 2,139 choices. "
     "Text search was only slightly better than random. The embedding difference "
     "turned out to be a weaker change signal than simple pixel difference. "
     "Brightness matching was the most useful step, reducing false alarms from "
     "24% to 9% of unchanged pixels. In the report we explain why we think this "
     "happened and what we would change next.")
P()
p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.15
r = p.add_run("Keywords – "); r.bold = True
r.font.name = "Times New Roman"; r.font.size = Pt(12)
r = p.add_run("Computer Vision, Remote Sensing, Satellite Imagery, Image "
              "Retrieval, Change Detection, CLIP, Embeddings, Vector Search.")
r.font.name = "Times New Roman"; r.font.size = Pt(12)

doc.add_page_break()

# ========================================================= TABLE OF CONTENTS
H1("Table of Contents")
def pg(key):
    return str(PAGES.get(key, ""))


_T = [("1.", "INTRODUCTION"), ("", "    1.1 Background"), ("", "    1.2 Motivation"),
      ("", "    1.3 Scope of the Project"),
      ("2.", "PROJECT DESCRIPTION AND GOALS"), ("", "    2.1 Literature Review"),
      ("", "    2.2 Research Gap"), ("", "    2.3 Objectives"),
      ("", "    2.4 Problem Statement"), ("", "    2.5 Project Plan"),
      ("3.", "TECHNICAL SPECIFICATION"), ("", "    3.1 Requirements"),
      ("", "        3.1.1 Functional"), ("", "        3.1.2 Non-Functional"),
      ("", "    3.2 Feasibility Study"), ("", "        3.2.1 Technical Feasibility"),
      ("", "        3.2.2 Economic Feasibility"), ("", "        3.2.3 Social Feasibility"),
      ("", "    3.3 System Specification"), ("", "        3.3.1 Hardware Specification"),
      ("", "        3.3.2 Software Specification"),
      ("4.", "DESIGN APPROACH AND DETAILS"), ("", "    4.1 System Architecture"),
      ("", "    4.2 Design"), ("", "        4.2.1 Data Flow Diagram"),
      ("", "        4.2.2 Use Case Diagram"),
      ("5.", "DATASET DETAILS"), ("", "    5.1 Why We Chose OSCD"),
      ("", "    5.2 What the Dataset Contains"), ("", "    5.3 The 24 City Pairs"),
      ("", "    5.4 Metadata We Extract"), ("", "    5.5 Limitations of the Dataset"),
      ("6.", "IMPLEMENTATION"), ("", "    6.1 Setup and Code Structure"),
      ("", "    6.2 Loading the Data and Extracting Metadata"),
      ("", "    6.3 Tiling"), ("", "    6.4 Embedding and the Shared Index"),
      ("", "    6.5 Search"), ("", "    6.6 Change Detection and False Alarm Removal"),
      ("", "    6.7 Problems We Faced"),
      ("7.", "RESULTS AND ANALYSIS"), ("", "    7.1 Experimental Setup"),
      ("", "    7.2 Text Search"), ("", "    7.3 Finding the Same Place Across Dates"),
      ("", "    7.4 Embeddings as a Change Signal"),
      ("", "    7.5 Pixel-Level Change Maps"), ("", "    7.6 False Alarm Removal"),
      ("", "    7.7 Summary and Future Work"),
      ("8.", "REFERENCES")]
toc = [("", "Abstract", "i")] + [(a, b, pg(b.strip())) for a, b in _T
]
t = doc.add_table(rows=1, cols=3)
t.style = "Table Grid"
for i, h in enumerate(["Sl.No", "Contents", "Page No."]):
    c = t.rows[0].cells[i]; c.text = ""
    r = c.paragraphs[0].add_run(h); r.bold = True
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
for sl, cont, pg in toc:
    cells = t.add_row().cells
    for i, v in enumerate([sl, cont, pg]):
        cells[i].text = ""
        pr = cells[i].paragraphs[0]; pr.paragraph_format.space_after = Pt(1)
        r = pr.add_run(v)
        r.font.name = "Times New Roman"; r.font.size = Pt(12)
        r.bold = bool(sl) and i == 1
for row in t.rows:
    row.cells[0].width = Inches(0.7)
    row.cells[1].width = Inches(4.4)
    row.cells[2].width = Inches(0.9)

from docx.oxml import OxmlElement


def _page_field(footer):
    p = footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    f = OxmlElement("w:fldSimple"); f.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r"); t_ = OxmlElement("w:t"); t_.text = "1"; r.append(t_); f.append(r)
    p._p.append(f)


def _numfmt(section, fmt, start):
    sp = section._sectPr
    for old in sp.findall(qn("w:pgNumType")):
        sp.remove(old)
    n = OxmlElement("w:pgNumType"); n.set(qn("w:fmt"), fmt); n.set(qn("w:start"), str(start))
    sp.find(qn("w:pgMar")).addnext(n)          # schema order: right after pgMar


front = doc.sections[0]
front.different_first_page_header_footer = True      # no number on the title page
_page_field(front.footer); _numfmt(front, "lowerRoman", 0)
main = doc.add_section(WD_SECTION.NEW_PAGE)
main.different_first_page_header_footer = False
main.footer.is_linked_to_previous = False
_page_field(main.footer); _numfmt(main, "decimal", 1)

# ============================================================= 1 INTRODUCTION
H1("1. Introduction")

H2("1.1 Background")
body("Earth observation satellites such as Sentinel-2 and Landsat photograph the "
     "whole planet again and again, and most of this data is free to download. "
     "Because of this, the amount of satellite imagery available today is far too "
     "large for anyone to look through by hand.")
body("The way these archives work right now is that you search using metadata. "
     "Metadata means the extra information stored along with the image, such as "
     "the date it was taken, the coordinates of the area, and which satellite "
     "took it. This is useful, but it has one big limitation: metadata does not "
     "describe what the picture actually shows. You cannot ask the archive to "
     "find \"a new bridge\" or \"houses built next to a river\", because the "
     "archive has no idea what is inside the image.")
body("There is a second problem as well. If someone wants to know how a place has "
     "changed over the last two years, they must find the old image, find the new "
     "image, line them up, and compare them visually. This takes a long time and "
     "it only works if the person already knows which place to check.")

H2("1.2 Motivation")
body("We chose this project because the problem is real and someone is actually "
     "asking for it. It comes from Smart India Hackathon 2026 problem statement "
     "SIH26227, which was given by the Ministry of Defence (Indian Army, DGIS). "
     "So this is not an imaginary problem we invented for marks.")
body("The same idea is useful in many other places too. City planners want to see "
     "where new construction is happening. Disaster teams want to see which "
     "buildings were damaged after a flood or an earthquake. Forest departments "
     "want to see where trees have been cut. All of these are the same basic "
     "question: what is here, and what changed?")
body("We were also motivated by one more thing. Most modern methods for this task "
     "use deep learning models that must be trained, and training needs labelled "
     "data, a good GPU, and a lot of time. We wanted to check whether we could get "
     "useful results using only models that are already trained by someone else. "
     "If that works, the system becomes cheap, fast to set up, and easy for anyone "
     "to run, even on a free Google Colab account.")

H2("1.3 Scope of the Project")
body("The official problem statement asks for six different features. We are four "
     "students working for about ten weeks, so we cannot build all six properly. "
     "Instead of promising everything and finishing nothing, we decided clearly "
     "which parts we will build and which parts we will leave out. We think being "
     "honest about this is better than claiming more than we did.")
body("We will fully build the search part and the change detection part. We will "
     "partly build the false alarm removal part, and we will clearly say which "
     "types of false alarms we handle. Grouping similar places together is a bonus "
     "feature that we will try only if time is left.")
body("We will not build the analyst review screen with an approve/reject history, "
     "and we will not build the offline server version that keeps adding new "
     "images without rebuilding the whole index. Those two are real engineering "
     "work for a company, not for a student project. We mention them as future "
     "work. We also only use free public datasets. No secret or military data is "
     "used anywhere in this project.")

doc.add_page_break()

# =============================================== 2 PROJECT DESCRIPTION & GOALS
H1("2. Project Description and Goals")

H2("2.1 Literature Review")
body("We read papers in three areas: models that connect images with text, methods "
     "that find change between two satellite images, and methods that search "
     "quickly through a large number of vectors. The table below shows the main "
     "papers we studied and what we learned from each one.")
table(["No.", "Paper and Year", "What they did", "What we learned / limitation"],
      [["1", "Radford et al., CLIP (2021)",
        "Trained one model on images and their text captions together, so both end "
        "up in the same number space.",
        "This is why we can search images by typing words. But it was trained on "
        "normal photos, not satellite photos."],
       ["2", "Liu et al., RemoteCLIP (2024)",
        "Took CLIP and trained it further on satellite and aerial images with text.",
        "The paper reports better results on satellite images than normal CLIP, so we "
        "tried it as our main model and compared it with normal CLIP ourselves."],
       ["3", "Oquab et al., DINOv2 (2023)",
        "Learned strong image features without using any labels.",
        "Good image features, but it cannot understand typed text, so it cannot do "
        "word search alone."],
       ["4", "Daudt et al., OSCD (2018)",
        "Used a Siamese network to compare two satellite images and find change. "
        "Also released the OSCD dataset.",
        "Shows that comparing features of two dates works well. But their model "
        "needs training with labelled data."],
       ["5", "Chen and Shi, LEVIR-CD (2020)",
        "Used attention to find building changes, and released a large change "
        "detection dataset.",
        "We use their dataset as our main dataset. Their method again needs "
        "supervised training."],
       ["6", "Malila, Change Vector Analysis (1980)",
        "Classic method. Subtracts the two images band by band and measures how "
        "big the difference is.",
        "Very simple and needs no training, so we use it as a baseline. It gets "
        "confused easily by lighting changes."],
       ["7", "Johnson et al., FAISS (2019)",
        "A library that can search through millions of vectors very quickly.",
        "This is how our search stays fast when the number of tiles grows."],
       ["8", "Recent EO foundation model work (2024–2026)",
        "Large pre-trained models are being applied to many satellite tasks.",
        "They transfer well, but almost every paper solves either search or change "
        "detection, not both."]],
      widths=[0.4, 1.5, 2.1, 2.1])
P()
body("One clear pattern came out of this reading. Searching and change detection "
     "are almost always treated as two separate systems with two separate sets of "
     "features. We found this strange, because both tasks are really asking "
     "\"how similar are these two things?\" That observation is where our project "
     "idea came from.")

H2("2.2 Research Gap")
body("From the papers above, we found these gaps:")
bullet("Search and change detection are built separately. Each one creates and "
       "stores its own features, which wastes memory and computing time.",
       bold_lead="Two separate systems. ")
bullet("Most change detection methods need labelled training data for the exact "
       "area they will be used on. Getting that labelled data is slow and "
       "expensive.", bold_lead="Training is almost always required. ")
bullet("Many papers report how much change they found (recall), but in real use "
       "the bigger problem is reporting change that is not actually there. If the "
       "system cries wolf too often, nobody will trust it.",
       bold_lead="Focus on finding change, not on being correct. ")
bullet("Season, cloud, shadow and sunlight differences are often not handled "
       "directly, even though they cause most of the wrong answers.",
       bold_lead="False alarms are not treated seriously. ")

H2("2.3 Objectives")
body("The main goals of this project are:")
numbered("To read satellite image tiles and keep their location and date "
         "information with them, so every result can be traced back to its source.")
numbered("To convert every tile into an embedding using a pre-trained model, "
         "without training anything ourselves.")
numbered("To store all embeddings in one single shared index that both search and "
         "change detection will use.")
numbered("To allow a user to search tiles by typing normal words, and also by "
         "giving an example image.")
numbered("To detect change between two dates by comparing the embeddings of the "
         "same tile, and to label the type of change.")
numbered("To remove false alarms caused by cloud, shadow and lighting differences, "
         "and to give a confidence value with every detected change.")
numbered("To measure the results properly using Precision@k, mAP, Precision, "
         "Recall, F1 and IoU, and to compare against simple baseline methods.")

H2("2.4 Problem Statement")
body("Satellite image archives can be searched only by metadata such as date, "
     "location and sensor. They cannot be searched by the content of the image, "
     "and they cannot directly answer what has changed in a place between two "
     "dates. Existing solutions treat these as two separate problems and build two "
     "separate systems, and most of them need labelled training data. On top of "
     "this, simple change detection methods produce too many false alarms because "
     "clouds, shadows, seasons and sunlight make unchanged places look different.")
body("The problem this project solves is: how can we make a satellite image "
     "archive searchable by meaning and by change over time, using one shared set "
     "of features, without training any model, while keeping the number of false "
     "alarms low?")

H2("2.5 Project Plan")
body("We planned the work over about ten weeks. The important point in our plan is "
     "that the four of us do not wait for each other. Each person can build and "
     "test their own part using temporary fake data, and only at the end do we join "
     "everything together. For example, the search module can be tested with random "
     "numbers instead of real embeddings, because the shape of the data is the same. "
     "This way nobody sits idle waiting for someone else to finish.")
figure(FIGS / "fig1_gantt.png", "Fig. 1. Gantt chart of the project plan")
body("The yellow bars in the middle are our checking stage. Before we go further, "
     "we run three tests that we call gates. Gate A checks that the search returns "
     "sensible tiles. Gate B checks that our change detection is better than simple "
     "image subtraction. Gate C checks that our false alarm removal actually "
     "removes false alarms without also deleting the real changes. If any gate "
     "fails, we fix that part before continuing.")

doc.add_page_break()

# ========================================================= 3 TECHNICAL SPEC
H1("3. Technical Specification")

H2("3.1 Requirements")

H3("3.1.1 Functional")
body("These are the things the system must be able to do:")
bullet("The system should read satellite images, including GeoTIFF files, and cut "
       "them into equal sized tiles.", bold_lead="Image loading and tiling: ")
bullet("Each tile must keep its coordinates, date and source file name so results "
       "can be traced back.", bold_lead="Keep location details: ")
bullet("The system should convert each tile into an embedding using a pre-trained "
       "model.", bold_lead="Make embeddings: ")
bullet("All embeddings must be stored in one index that both search and change "
       "detection read from.", bold_lead="Build one shared index: ")
bullet("A user should be able to type words and get back a ranked list of matching "
       "tiles.", bold_lead="Text search: ")
bullet("A user should be able to pick one tile and ask for other tiles that look "
       "similar.", bold_lead="Image search: ")
bullet("Results should be filterable by date range, area and sensor.",
       bold_lead="Filters: ")
bullet("For the same tile at two dates, the system must produce a change map "
       "showing which pixels changed.", bold_lead="Change detection: ")
bullet("Each changed region should be labelled as construction, clearance, water "
       "or vegetation change.", bold_lead="Change type: ")
bullet("Changes caused by cloud, shadow or lighting should be removed, and a "
       "confidence score given.", bold_lead="False alarm removal: ")

H3("3.1.2 Non-Functional")
bullet("A search should return results in under one second for a few thousand "
       "tiles.", bold_lead="Speed: ")
bullet("The whole system must run on a free Google Colab account without a paid "
       "GPU.", bold_lead="Low cost: ")
bullet("Running the same input twice must give the same output, so our results can "
       "be checked by others.", bold_lead="Repeatable: ")
bullet("If FAISS or rasterio is not installed, the system should fall back to a "
       "simpler method instead of crashing.", bold_lead="Does not break easily: ")
bullet("The code is split into small files owned by one person each, so we do not "
       "overwrite each other's work.", bold_lead="Easy to maintain: ")
bullet("The name, version and licence of every pre-trained model must be written "
       "down in the report.", bold_lead="Honest about models used: ")
bullet("Only free and public datasets are used. No restricted data.",
       bold_lead="Data safety: ")

H2("3.2 Feasibility Study")

H3("3.2.1 Technical Feasibility")
body("This project is technically possible for us. The models we need, such as "
     "CLIP and RemoteCLIP, are already trained and free to download, so we do not "
     "need a powerful computer for training. The datasets, LEVIR-CD and OSCD, are "
     "free and already contain the correct answers, so we can measure our results "
     "properly. One important detail is that both datasets give image pairs that "
     "are already aligned on top of each other. Aligning two satellite images taken "
     "from different angles is a hard problem by itself, and choosing these "
     "datasets lets us avoid it. All the libraries we need, such as OpenCV, NumPy "
     "and PyTorch, are free and well documented.")

H3("3.2.2 Economic Feasibility")
body("The cost of this project is almost zero. The datasets are free, the models "
     "are free, and the libraries are open source. We use Google Colab's free tier "
     "for computing, so we do not need to buy a GPU. The only real cost is our own "
     "time and some disk space, because the LEVIR-CD dataset is around 3.8 GB. For "
     "a real organisation, the saving is also clear: because no training is needed, "
     "they do not have to pay people to label thousands of images.")

H3("3.2.3 Social Feasibility")
body("The system is meant to help a human analyst, not replace one. The person "
     "still makes the final decision; our system only shortens the list of places "
     "they need to look at. This matters, because an automatic system that quietly "
     "makes mistakes could be harmful. That is why every detected change comes with "
     "a confidence score and a link back to the original image, so the analyst can "
     "always check it themselves.")
body("We also thought about misuse. Satellite monitoring can affect privacy, so we "
     "work only at a scale where buildings and roads are visible, not individual "
     "people. All our data is public and openly licensed.")

H2("3.3 System Specification")

H3("3.3.1 Hardware Specification")
table(["Component", "Minimum requirement", "Why it is needed"],
      [["Processor", "Intel i5 or equivalent (4 cores)",
        "To run tiling and image processing at a reasonable speed."],
       ["Memory (RAM)", "8 GB (16 GB better)",
        "Large satellite images take a lot of memory while being tiled."],
       ["Storage", "20 GB free space",
        "LEVIR-CD alone is about 3.8 GB, plus models and saved results."],
       ["GPU", "Optional (Colab free T4 is enough)",
        "Only makes embedding faster. The system still works on CPU."],
       ["Internet", "Required once",
        "To download the datasets and the pre-trained model weights."]],
      widths=[1.3, 2.0, 2.8])

P()
H3("3.3.2 Software Specification")
table(["Software", "Version", "What we use it for"],
      [["Operating System", "Windows 11 / Ubuntu 22.04",
        "Any OS works; we developed on both."],
       ["Python", "3.10 or above", "Main programming language."],
       ["Google Colab", "Free tier", "Running the notebooks without local setup."],
       ["PyTorch", "2.0+", "Running the pre-trained model (prediction only)."],
       ["open_clip_torch", "2.24+", "Loading CLIP and RemoteCLIP models."],
       ["FAISS", "1.7+", "Fast searching through all the embeddings."],
       ["OpenCV", "4.8+", "Reading images, masks, filtering and morphology."],
       ["NumPy", "1.24+", "All the number and vector calculations."],
       ["rasterio", "1.3+", "Reading GeoTIFF files and their coordinates."],
       ["scikit-learn", "1.3+", "Clustering similar places and some metrics."],
       ["Matplotlib", "3.7+", "Drawing the result figures and charts."]],
      widths=[1.5, 1.2, 3.4])

doc.add_page_break()

# ======================================================= 4 DESIGN APPROACH
H1("4. Design Approach and Details")

H2("4.1 System Architecture")
body("The design of our system is shown in Fig. 2. The most important idea in the "
     "whole project is the orange box in the middle, the shared index. Normally a "
     "system that searches images and a system that finds change would each build "
     "their own features. In our design we build the features only once and let "
     "both sides use the same ones.")
figure(FIGS / "fig2_architecture.png", "Fig. 2. System architecture")
body("Each step is explained below, along with the reason we designed it that way.")
bullet("A full satellite image is too big to handle at once, and we also want "
       "results that point to a specific spot rather than the whole scene. So we "
       "cut each image into small square tiles. We store the coordinates and the "
       "date with every tile, because a result that says \"something changed\" is "
       "useless if you cannot say where.", bold_lead="Step 1 – Cut into tiles. ")
bullet("Each tile is passed through a pre-trained CLIP or RemoteCLIP model, which "
       "turns the picture into a list of about 512 numbers. Tiles that look alike "
       "get similar numbers. We freeze this model, meaning we do not train it at "
       "all, we only use it to make predictions. We chose RemoteCLIP because normal "
       "CLIP was trained on everyday photos and does not understand satellite views "
       "as well.", bold_lead="Step 2 – Make embeddings. ")
bullet("All these number lists are stored together in one index using FAISS. This "
       "is the heart of our idea. Because the numbers describe what a tile looks "
       "like, the same numbers can answer \"find me tiles like this\" and \"did "
       "this tile change?\". Building them once saves both memory and time.",
       bold_lead="Shared index. ")
bullet("When a user types words, the same CLIP model turns those words into a list "
       "of numbers in the same space. We then compare the query numbers with every "
       "tile's numbers and sort the tiles by how close they are. The user can also "
       "give an example tile instead of words.",
       bold_lead="Step 3 – Search. ")
bullet("For one place at two dates, we take the two embeddings of that tile and "
       "measure the distance between them. A small distance means nothing much "
       "happened. A large distance means something changed. After that we compare "
       "the pixels to find exactly which part changed, and we label the change as "
       "construction, clearance, water or vegetation.",
       bold_lead="Step 4 – Find change. ")
bullet("This step exists because two images of the same unchanged place can still "
       "look very different. A cloud may be in the way, the sun may be at a "
       "different angle, or the season may have changed the colour of the plants. "
       "We first match the brightness of the two images, then mask out clouds and "
       "shadows, and finally give each detected change a confidence score. We prefer "
       "to report fewer changes that are correct rather than many changes that are "
       "wrong.", bold_lead="Step 5 – Remove false alarms. ")
bullet("Finally we measure everything. For search we use Precision@k and mAP, which "
       "tell us how many of the top results were actually relevant. For change we "
       "use Precision, Recall, F1 and IoU, which compare our change map with the "
       "correct answer given in the dataset. We also record speed and storage used.",
       bold_lead="Step 6 – Measure. ")

doc.add_page_break()

H2("4.2 Design")

H3("4.2.1 Data Flow Diagram")
body("Fig. 3 shows how the data moves through our system. We have used the standard "
     "symbols: squares are things outside the system, rounded boxes are the "
     "processes inside it, and the open box is where data is stored.")
figure(FIGS / "fig3_dfd.png", "Fig. 3. Level-1 Data Flow Diagram")
body("The bottom row is the preparation path, which runs once. The satellite image "
     "files go into process 1.0, which cuts them into tiles. Those tiles go into "
     "process 2.0, which turns them into embeddings and saves them in the data "
     "store D1.")
body("The top row is what happens when someone actually uses the system. If the "
     "user types a query, it goes to process 3.0, which reads D1 and sends back a "
     "ranked list of tiles. If the user wants to compare two dates, process 4.0 "
     "reads the same D1 and produces a rough change result, which is cleaned by "
     "process 5.0 before being shown to the user.")
body("The thing to notice in this diagram is that D1 has two arrows coming out of "
     "it, one going to search and one going to change detection. That single shared "
     "store is the main point of our design.")

H3("4.2.2 Use Case Diagram")
body("Fig. 4 shows who uses the system and what they can do with it. We have two "
     "types of users. The Analyst is the main user who searches and looks at "
     "changes. The Data Admin is the person who loads new satellite images and "
     "builds the index.")
figure(FIGS / "fig4_usecase.png", "Fig. 4. Use Case Diagram", width=5.9)
body("The Analyst can search by typing words, search using an example image, narrow "
     "down results using date, area or sensor filters, compare two dates to see what "
     "changed, and look at the change type along with its confidence value. The "
     "Data Admin loads the images and builds the index. Both of them can export "
     "results, and every exported result carries the source image name and date "
     "with it, so the information can always be checked later.")

doc.add_page_break()

import new_sections
new_sections.add(dict(P=P, H1=H1, H2=H2, H3=H3, body=body, bullet=bullet,
                      numbered=numbered, table=table, doc=doc, RFIG=RFIG,
                      ROOT=ROOT, Pt=Pt, Inches=Inches, qn=qn,
                      WD_ALIGN_PARAGRAPH=WD_ALIGN_PARAGRAPH))

# ============================================================== 8 REFERENCES
H1("8. References")

P("Journals and Conferences: <IEEE Format>", size=12, bold=True, spacing=1.15)
refs = [
 "A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, "
 "A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, \"Learning "
 "transferable visual models from natural language supervision,\" in Proc. 38th "
 "Int. Conf. Machine Learning (ICML), 2021, pp. 8748–8763.",

 "F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou, "
 "\"RemoteCLIP: A vision language foundation model for remote sensing,\" IEEE "
 "Trans. Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.",

 "M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, "
 "P. Fernandez, D. Haziza, F. Massa, and A. El-Nouby, \"DINOv2: Learning robust "
 "visual features without supervision,\" Transactions on Machine Learning "
 "Research, 2024.",

 "R. C. Daudt, B. Le Saux, A. Boulch, and Y. Gousseau, \"Urban change detection "
 "for multispectral earth observation using convolutional neural networks,\" in "
 "Proc. IEEE Int. Geoscience and Remote Sensing Symp. (IGARSS), 2018, "
 "pp. 2115–2118.",

 "H. Chen and Z. Shi, \"A spatial-temporal attention-based method and a new "
 "dataset for remote sensing image change detection,\" Remote Sensing, vol. 12, "
 "no. 10, p. 1662, 2020.",

 "W. A. Malila, \"Change vector analysis: An approach for detecting forest "
 "changes with Landsat,\" in Proc. LARS Symp. Machine Processing of Remotely "
 "Sensed Data, 1980, pp. 326–335.",

 "J. Johnson, M. Douze, and H. Jegou, \"Billion-scale similarity search with "
 "GPUs,\" IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2019.",

 "Z. Zheng, Y. Zhong, J. Wang, A. Ma, and L. Zhang, \"Building damage assessment "
 "for rapid disaster response with a deep object-based semantic change detection "
 "framework,\" Remote Sensing of Environment, vol. 265, p. 112636, 2021.",

 "Y. Zhan, K. Fu, M. Yan, X. Sun, H. Wang, and X. Qiu, \"Change detection based "
 "on deep Siamese convolutional network for optical aerial images,\" IEEE "
 "Geoscience and Remote Sensing Letters, vol. 14, no. 10, pp. 1845–1849, 2017.",

 "S. Lobry, D. Marcos, J. Murray, and D. Tuia, \"RSVQA: Visual question answering "
 "for remote sensing data,\" IEEE Trans. Geoscience and Remote Sensing, vol. 58, "
 "no. 12, pp. 8555–8566, 2020.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)
    run = p.add_run(f"[{i}]  {r}")
    run.font.name = "Times New Roman"; run.font.size = Pt(12)

P()
P("Weblinks:", size=12, bold=True, spacing=1.15)
links = [
 "LEVIR-CD dataset – https://justchenhao.github.io/LEVIR/",
 "OSCD (Onera Satellite Change Detection) dataset – https://rcdaudt.github.io/oscd/",
 "Copernicus Data Space Ecosystem (Sentinel-1/2) – https://dataspace.copernicus.eu/",
 "USGS EarthExplorer (Landsat Collection 2) – https://earthexplorer.usgs.gov/",
 "NRSC / ISRO Bhuvan Open Data Archive – https://bhuvan-app3.nrsc.gov.in/data/download/",
 "Smart India Hackathon 2026, Problem Statement SIH26227 – https://www.sih.gov.in/",
]
for l in links:
    bullet(l)

doc.save(OUT)
print("saved:", OUT)
