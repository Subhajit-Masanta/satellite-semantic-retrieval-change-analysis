"""Chapters 1-4 of the report, written in a casual student voice.
Working copy for the team; facts and numbers are the real ones."""


def abstract(h):
    body = h["body"]
    body("Satellites are taking pictures of the Earth every day, so now there is a very "
         "huge amount of satellite images stored. But the problem is we can search "
         "these images only by date, coordinates or satellite name. We cannot search by "
         "what is there inside the picture, so the person should already know where to "
         "look. And to find what changed in a place is even more hard.")
    body("So in our project we tried to solve both the problems together. First we cut "
         "every image into small tiles. Then each tile we give to a pre-trained CLIP "
         "model, and it gives back a list of numbers which is called embedding. All the "
         "embeddings we store one time only, in one shared index, and we use it for two "
         "works. First is search, where user can type words like \"a river\" or give one "
         "example tile and our system gives the matching tiles. Second is change, where "
         "for same place at two dates we compare the two embeddings. We also try to "
         "remove the false alarms which come from brightness difference, cloud and "
         "shadow. We did not train any model.")
    body("We tested on three datasets: OSCD (24 pairs of Sentinel-2 city images at "
         "10 m), LEVIR-CD (building change at 0.5 m) and EuroSAT (27,000 labelled "
         "patches, for search). Search worked: on OSCD the correct tile of the same "
         "place came first 46% of the time out of 2,139 tiles, and on EuroSAT 69% of "
         "the first 10 text search results were correct. For change, our idea worked "
         "on the sharp LEVIR-CD images (AUC 0.92, pixel difference gets 0.70) but it "
         "was weak on OSCD (AUC 0.70), where a new building is only few pixels. "
         "Brightness matching reduced false alarms from 24% to 9% of unchanged "
         "pixels.")


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
         "leave.")
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
         "detection. In them we found these gaps, and our proposed project is "
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
    body("The yellow bars in the middle are our testing stage. Before joining "
         "everything we test each part separately, and if something is not working "
         "we fix that part first and then continue.")

    doc.add_page_break()

    # ---------------------------------------------------------------- 4
    H1("3. Design Approach and Details")

    H2("3.1 System Architecture")
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
           "of no use if we cannot say where.", bold_lead="Step 1: Cut into tiles. ")
    bullet("Each tile goes through a pre-trained CLIP or RemoteCLIP model, which "
           "turns the picture into a list of 512 numbers. Tiles which look alike get "
           "similar numbers. We freeze the model, means we do not train it at all, we "
           "only use it for prediction. We tried RemoteCLIP because normal CLIP was "
           "trained on everyday photos and not on satellite views.",
           bold_lead="Step 2: Make embeddings. ")
    bullet("All these number lists we store together in one index. This is the heart "
           "of our idea. Since the numbers describe how a tile looks, the same "
           "numbers can answer \"find tiles like this\" and also \"did this tile "
           "change?\". Making them once saves memory and time both.",
           bold_lead="Shared index. ")
    bullet("When user types words, the same CLIP model turns those words into numbers "
           "in the same space. Then we compare the query numbers with every tile's "
           "numbers and sort the tiles by how close they are. User can also give an "
           "example tile in place of words.", bold_lead="Step 3: Search. ")
    bullet("For one place at two dates, we take the two embeddings of that tile and "
           "measure the distance between them. Small distance means nothing much "
           "happened, large distance means something changed. After that we compare "
           "the pixels to find exactly which part changed, and we label the change as "
           "construction, clearance, water or vegetation.",
           bold_lead="Step 4: Find change. ")
    bullet("This step is there because two images of same unchanged place can still "
           "look very different. A cloud may be in between, sun may be at different "
           "angle, or season may have changed the colour of plants. First we match "
           "the brightness of both images, then mask out cloud and shadow, and lastly "
           "give each change a confidence score. We prefer to report less changes "
           "which are correct than many changes which are wrong.",
           bold_lead="Step 5: Remove false alarms. ")
    bullet("At last we measure everything. For search we use Precision@k and mAP, "
           "which tell how many of the top results were actually relevant. For change "
           "we use Precision, Recall, F1 and IoU, which compare our change map with "
           "the correct answer given in the dataset. We also note the speed and "
           "storage used.", bold_lead="Step 6: Measure. ")

    doc.add_page_break()
