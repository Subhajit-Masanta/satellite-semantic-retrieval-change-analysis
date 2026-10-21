"""Chapters 1-4 of the report, written in a casual student voice.
Working copy for the team; facts and numbers are the real ones."""


def abstract(h):
    body = h["body"]
    body("Satellites are taking pictures of the Earth every day, so now there is a very "
         "huge amount of satellite images stored. But the problem is we can search "
         "these images only by date, coordinates or satellite name. We cannot search by "
         "what is there inside the picture, so the person should already know where to "
         "look. And to find what changed in a place is even more hard, because somebody "
         "has to open two images and compare by eye.")
    body("So in our project we tried to solve both the problems together. First we cut "
         "every image into small tiles. Then each tile we give to a pre-trained CLIP "
         "model, and it gives back a list of numbers which is called embedding. All the "
         "embeddings we store one time only, in one shared index, and we use it for two "
         "works. First is search, where user can type words like \"a river\" or give one "
         "example tile and our system gives the matching tiles. Second is change, where "
         "for same place at two dates we compare the two embeddings. We also try to "
         "remove the false alarms which come from brightness difference, cloud and "
         "shadow. We did not train any model.")
    body("We tested on OSCD dataset which is having 24 pairs of Sentinel-2 city images "
         "with correct change masks. Our results are mixed and we are writing them as "
         "it is. Searching the same place across two dates worked good, the correct "
         "tile came first 46% of the time out of 2,139 tiles. Text search was only "
         "little better than random. The embedding difference came out weaker than "
         "simple pixel difference for finding change. Brightness matching was the most "
         "useful step, it reduced false alarms from 24% to 9% of unchanged pixels.")


def add(h):
    P, H1, H2, H3, body, bullet, numbered, table, figure, doc, FIGS = (h[k] for k in (
        "P", "H1", "H2", "H3", "body", "bullet", "numbered", "table", "figure", "doc", "FIGS"))

    # ---------------------------------------------------------------- 1
    H1("1. Introduction")

    H2("1.1 Background")
    body("Satellites like Sentinel-2 and Landsat take photos of the whole Earth again "
         "and again, and most of this data is free to download. Because of this the "
         "amount of satellite images today is so much that nobody can check them by "
         "hand.")
    body("Right now these archives are searched using metadata. Metadata is the extra "
         "information saved with the image, like which date it was taken, the "
         "coordinates of the area and which satellite took it. This is useful but it "
         "has one big limitation, metadata does not tell what the picture is actually "
         "showing. We cannot ask the archive to find \"a new bridge\" or \"houses near a "
         "river\", because the archive does not know what is inside the image.")
    body("One more problem is also there. If someone wants to know how a place changed "
         "in last two years, they have to find the old image, find the new image, "
         "keep them one on other and compare by seeing. This takes lot of time and it "
         "works only if the person already knows which place to check.")

    H2("1.2 Motivation")
    body("We took this project because the problem is real and somebody is actually "
         "asking for it. It is from Smart India Hackathon 2026, problem statement "
         "SIH26227, given by Ministry of Defence (Indian Army, DGIS). So it is not "
         "some problem we made up just for marks.")
    body("Same idea is useful in many other places also. City planners want to see "
         "where new construction is coming. Disaster teams want to see which buildings "
         "got damaged after flood or earthquake. Forest department wants to see where "
         "trees are cut. Basically all are asking the same question, what is here and "
         "what changed.")
    body("One more thing motivated us. Most of the new methods for this use deep "
         "learning models which have to be trained, and training needs labelled data, "
         "good GPU and lot of time. We wanted to check if we can get useful results "
         "using only the models which somebody already trained. If it works then the "
         "system becomes cheap, fast to setup and anybody can run it, even on a normal "
         "laptop.")

    H2("1.3 Scope of the Project")
    body("The official problem statement is asking for six different features. We are "
         "four students and we had around ten weeks, so building all six properly was "
         "not possible for us. So instead of promising everything and finishing "
         "nothing, we decided clearly which parts we will build and which we will "
         "leave. We feel it is better to be honest here than to claim more.")
    body("The search part and the change detection part we built fully. The false "
         "alarm removal part we built partly, and we tell clearly which type of false "
         "alarms we handle. Grouping similar places together is a bonus feature, we "
         "will try only if time is left.")
    body("We did not build the analyst review screen with approve/reject history, and "
         "we did not build the offline server version which keeps adding new images "
         "without rebuilding whole index. These two are proper engineering work for a "
         "company, not for a student project, so we kept them as future work. Also we "
         "used only free public datasets. No secret or military data is used anywhere "
         "in our project.")

    doc.add_page_break()

    # ---------------------------------------------------------------- 2
    H1("2. Project Description and Goals")

    H2("2.1 Research Gap")
    body("We looked how the existing systems are doing image search and change "
         "detection [1]-[7]. In them we found these gaps, and our proposed project is "
         "trying to work on these:")
    bullet("Search and change detection are built separately. Each one makes and "
           "stores its own features, so memory and computing time both get wasted.",
           bold_lead="Two separate systems. ")
    bullet("Most change detection methods need labelled training data of the same "
           "area where they will be used. Getting that labelled data is slow and "
           "costly.", bold_lead="Training is needed almost always. ")
    bullet("Many papers tell how much change they found (recall). But in real use the "
           "bigger problem is telling a change which is not there. If the system "
           "gives wrong alarm too many times, nobody will trust it.",
           bold_lead="They focus on finding change, not on being correct. ")
    bullet("Season, cloud, shadow and sunlight differences are mostly not handled "
           "directly, even though these only cause most of the wrong answers.",
           bold_lead="False alarms are not taken seriously. ")

    H2("2.2 Objectives")
    body("In our proposed project we want to do these things:")
    numbered("To read the satellite image tiles and keep their location and date "
             "with them, so that every result we can trace back to where it came from.")
    numbered("To convert every tile into an embedding using a pre-trained model, "
             "without training anything by ourself.")
    numbered("To store all the embeddings in one single shared index, which both "
             "search and change detection will use.")
    numbered("To let the user search tiles by typing normal words, and also by giving "
             "an example image.")
    numbered("To detect change between two dates by comparing the embeddings of the "
             "same tile, and to tell the type of change.")
    numbered("To remove false alarms coming from cloud, shadow and lighting "
             "difference, and to give a confidence value with every change.")
    numbered("To measure the results properly using Precision@k, mAP, Precision, "
             "Recall, F1 and IoU, and compare with simple baseline methods.")

    H2("2.3 Problem Statement")
    body("Satellite image archives can be searched only by metadata like date, "
         "location and sensor. They cannot be searched by content of the image, and "
         "they cannot directly tell what changed in a place between two dates. The "
         "existing solutions take these as two separate problems and make two "
         "separate systems, and most of them need labelled training data. On top of "
         "this, the simple change detection methods give too many false alarms, "
         "because cloud, shadow, season and sunlight make the unchanged places also "
         "look different.")
    body("So the problem we take in our proposed project is: how can we make a "
         "satellite image archive searchable by meaning and by change over time, "
         "using one shared set of features, without training any model, and still "
         "keeping the false alarms low?")

    H2("2.4 Project Plan")
    body("We planned the work for around ten weeks. The main point of our plan is "
         "that four of us do not wait for each other. Each person builds and tests "
         "their own part using temporary fake data, and only at the end we join "
         "everything. For example the search part can be tested with random numbers "
         "in place of real embeddings, because shape of the data is same. Like this "
         "nobody is sitting idle waiting for other person to finish.")
    figure(FIGS / "fig1_gantt.png", "Fig. 1. Gantt chart of the project plan")
    body("The yellow bars in the middle are our checking stage. Before going further "
         "we run three tests which we call as gates. Gate A checks that search is "
         "giving sensible tiles. Gate B checks that our change detection is better "
         "than simple image subtraction. Gate C checks that our false alarm removal "
         "is really removing false alarms and not deleting the real changes also. If "
         "any gate fails, we fix that part first and then continue.")

    doc.add_page_break()

    # ---------------------------------------------------------------- 3
    H1("3. Technical Specification")
    H2("3.1 Requirements")

    H3("3.1.1 Functional")
    body("This is what our proposed system is doing, step by step:")
    bullet("We read the Sentinel-2 band files (GeoTIFF, 16-bit) and cut every scene "
           "into tiles of 64 x 64 px with no overlap.",
           bold_lead="Image loading and tiling: ")
    bullet("For every tile we store its pixel offset (x, y) inside the scene, the "
           "tile size, the acquisition date and the scene name. Position of the scene "
           "we store as longitude/latitude bounds [min_lon, min_lat, max_lon, max_lat] "
           "which we read from the .geojson file, for example [-48.0498, -15.8596, "
           "-48.0016, -15.8164] for aguasclaras.", bold_lead="Location details: ")
    bullet("Each tile we pass through a frozen CLIP ViT-B/32 model and we get a "
           "vector of 512 values, which we scale to length 1.",
           bold_lead="Embeddings: ")
    bullet("All vectors we keep in one array and the metadata in a list of same "
           "order. Search and change detection both are reading this same index.",
           bold_lead="One shared index: ")
    bullet("We encode the typed query with CLIP text encoder and rank all tiles by "
           "cosine similarity.", bold_lead="Text search: ")
    bullet("We take the vector of a tile which user picked and return its nearest "
           "tiles.", bold_lead="Image search: ")
    bullet("We can cut down the ranked list by date range, by sensor, and by an area "
           "box given as longitude/latitude bounds.", bold_lead="Filters: ")
    bullet("For same tile at two dates we compute the distance between its two "
           "vectors and also one pixel change mask.", bold_lead="Change detection: ")
    bullet("Each changed region we label as construction, clearance, water or "
           "vegetation using colour, texture and shape rules.",
           bold_lead="Change type: ")
    bullet("We match the brightness of date 2 to date 1, mask the cloud and shadow, "
           "and give a confidence value with every change.",
           bold_lead="False alarm removal: ")

    H3("3.1.2 Non-Functional")
    bullet("One search over our 4,278 tiles took about 0.32 ms in our run.",
           bold_lead="Speed: ")
    bullet("We ran the full OSCD experiment on laptop CPU (Intel i5, 16 GB RAM) in "
           "about 6 minutes, no paid GPU.", bold_lead="Low cost: ")
    bullet("We fix the random seed and save every number in a results file, so same "
           "input gives same output.", bold_lead="Repeatable: ")
    bullet("If FAISS or rasterio is not installed, our code shifts to NumPy search "
           "and OpenCV reading and it does not crash.",
           bold_lead="Does not break easily: ")
    bullet("We split the code in small files, one owner for each, so we do not "
           "overwrite each other's work.", bold_lead="Easy to maintain: ")
    bullet("We note down the name, source and licence of every pre-trained model we "
           "use.", bold_lead="Model details: ")
    bullet("We use only free and public datasets. No restricted data.",
           bold_lead="Data safety: ")

    H2("3.2 Feasibility Study")

    H3("3.2.1 Technical Feasibility")
    body("Our proposed project is technically possible for us. CLIP and RemoteCLIP "
         "models are already trained and free to download, so we are not training "
         "anything and we do not need a strong computer. The OSCD dataset which we "
         "used is free, around 513 MB, and it already has the correct change masks, "
         "so we can measure our results in numbers. The two dates of every OSCD city "
         "are already aligned pixel to pixel. Aligning two satellite images by "
         "ourself is a hard problem, and by taking this dataset we avoid it. OpenCV, "
         "NumPy and PyTorch are free and well documented.")

    H3("3.2.2 Economic Feasibility")
    body("Cost of our proposed project is almost zero. Dataset, models and libraries "
         "all are free. We ran everything on our own laptop, so we did not rent any "
         "GPU or cloud machine. The only real cost is our time and around 3 GB disk "
         "space for the dataset and model weights. For a real organisation one more "
         "saving is there, since nothing is trained nobody has to label thousands of "
         "images.")

    H3("3.2.3 Social Feasibility")
    body("Our proposed system is made to help a human analyst, not to replace. The "
         "person still takes the final decision, our system only makes the list of "
         "places shorter. This matters because an automatic system which makes "
         "mistakes silently can be harmful. That is why with every detected change we "
         "give a confidence score and the link back to original image, so analyst can "
         "always check it by themself.")
    body("We also thought about misuse. Satellite monitoring can affect privacy, so "
         "we work only at the scale where buildings and roads are visible, not "
         "individual people. All our data is public and openly licensed.")

    H2("3.3 System Specification")

    H3("3.3.1 Hardware Specification")
    table(["Component", "Minimum requirement", "Why it is needed"],
          [["Processor", "Intel i5 or equivalent (4 cores)",
            "To run tiling and image processing in reasonable speed."],
           ["Memory (RAM)", "8 GB (16 GB better)",
            "Big satellite images take lot of memory while tiling."],
           ["Storage", "20 GB free space",
            "OSCD is about 1 GB unzipped, model weights about 1.2 GB, plus results."],
           ["GPU", "Optional",
            "Only makes embedding faster. We ran our experiments on CPU."],
           ["Internet", "Required once",
            "To download the dataset and the pre-trained model weights."]],
          widths=[1.3, 2.0, 2.8])
    P()
    H3("3.3.2 Software Specification")
    table(["Software", "Version", "What we use it for"],
          [["Operating System", "Windows 11", "We developed and ran on this."],
           ["Python", "3.10 or above", "Main programming language."],
           ["PyTorch", "2.0+", "Running the pre-trained model (prediction only)."],
           ["open_clip_torch", "2.24+", "Loading CLIP and RemoteCLIP models."],
           ["FAISS", "1.7+", "Fast search in embeddings (optional, NumPy used if missing)."],
           ["OpenCV", "4.8+", "Reading images, masks, filtering and morphology."],
           ["NumPy", "1.24+", "All number and vector calculations."],
           ["rasterio", "1.3+", "Reading GeoTIFF files and their header."],
           ["scikit-learn", "1.3+", "Clustering similar places and some metrics."],
           ["Matplotlib", "3.7+", "Drawing the result figures and charts."]],
          widths=[1.5, 1.2, 3.4])

    doc.add_page_break()

    # ---------------------------------------------------------------- 4
    H1("4. Design Approach and Details")

    H2("4.1 System Architecture")
    body("The design of our proposed system is shown in Fig. 2. The most important "
         "idea of the whole project is the orange box in the middle, the shared "
         "index. Normally a system which searches images and a system which finds "
         "change will each build their own features. In our design we build the "
         "features one time only and both sides use the same.")
    figure(FIGS / "fig2_architecture.png", "Fig. 2. System architecture")
    body("Each step is explained below, with the reason why we did it like that.")
    bullet("One full satellite image is too big to handle at once, and also we want "
           "the result to point to a small spot and not the whole scene. So we cut "
           "each image into small square tiles. With every tile we keep the "
           "coordinates and date, because a result which says \"something changed\" is "
           "of no use if we cannot say where.", bold_lead="Step 1 – Cut into tiles. ")
    bullet("Each tile goes through a pre-trained CLIP or RemoteCLIP model, which "
           "turns the picture into a list of 512 numbers. Tiles which look alike get "
           "similar numbers. We freeze the model, means we do not train it at all, we "
           "only use it for prediction. We tried RemoteCLIP because normal CLIP was "
           "trained on everyday photos and not on satellite views.",
           bold_lead="Step 2 – Make embeddings. ")
    bullet("All these number lists we store together in one index. This is the heart "
           "of our idea. Since the numbers describe how a tile looks, the same "
           "numbers can answer \"find tiles like this\" and also \"did this tile "
           "change?\". Making them once saves memory and time both.",
           bold_lead="Shared index. ")
    bullet("When user types words, the same CLIP model turns those words into numbers "
           "in the same space. Then we compare the query numbers with every tile's "
           "numbers and sort the tiles by how close they are. User can also give an "
           "example tile in place of words.", bold_lead="Step 3 – Search. ")
    bullet("For one place at two dates, we take the two embeddings of that tile and "
           "measure the distance between them. Small distance means nothing much "
           "happened, large distance means something changed. After that we compare "
           "the pixels to find exactly which part changed, and we label the change as "
           "construction, clearance, water or vegetation.",
           bold_lead="Step 4 – Find change. ")
    bullet("This step is there because two images of same unchanged place can still "
           "look very different. A cloud may be in between, sun may be at different "
           "angle, or season may have changed the colour of plants. First we match "
           "the brightness of both images, then mask out cloud and shadow, and lastly "
           "give each change a confidence score. We prefer to report less changes "
           "which are correct than many changes which are wrong.",
           bold_lead="Step 5 – Remove false alarms. ")
    bullet("At last we measure everything. For search we use Precision@k and mAP, "
           "which tell how many of the top results were actually relevant. For change "
           "we use Precision, Recall, F1 and IoU, which compare our change map with "
           "the correct answer given in the dataset. We also note the speed and "
           "storage used.", bold_lead="Step 6 – Measure. ")

    doc.add_page_break()

    H2("4.2 Design")

    H3("4.2.1 Data Flow Diagram")
    body("Fig. 3 shows how the data is moving in our system. We used the standard "
         "symbols: squares are things outside the system, rounded boxes are the "
         "processes inside, and the open box is where data is stored.")
    figure(FIGS / "fig3_dfd.png", "Fig. 3. Level-1 Data Flow Diagram")
    body("The bottom row is the preparation path and it runs only once. The satellite "
         "image files go into process 1.0 which cuts them into tiles. Those tiles go "
         "to process 2.0 which turns them into embeddings and saves in data store D1.")
    body("The top row is what happens when somebody actually uses the system. If user "
         "types a query, it goes to process 3.0, which reads D1 and sends back a "
         "ranked list of tiles. If user wants to compare two dates, process 4.0 reads "
         "the same D1 and gives a rough change result, and process 5.0 cleans it "
         "before showing to the user.")
    body("The thing to notice in this diagram is that D1 has two arrows going out, "
         "one to search and one to change detection. That single shared store is the "
         "main point of our design.")

    H3("4.2.2 Use Case Diagram")
    body("Fig. 4 shows who uses our proposed system and what they can do with it. We "
         "have two types of users. Analyst is the main user who searches and looks at "
         "the changes. Data Admin is the person who loads new satellite images and "
         "builds the index.")
    figure(FIGS / "fig4_usecase.png", "Fig. 4. Use Case Diagram", width=5.9)
    body("The Analyst can search by typing words, search by an example image, narrow "
         "the results using date, area or sensor filters, compare two dates to see "
         "what changed, and see the change type with its confidence value. The Data "
         "Admin loads the images and builds the index. Both of them can export "
         "results, and every exported result carries the source image name and date "
         "with it, so the information can always be checked later.")

    doc.add_page_break()
