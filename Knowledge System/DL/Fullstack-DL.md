# 1. Setting up Machine Learning Projects

## 	a. Overview:

- ML is still research, therefore it is very challenging to aim for 100% success rate.

- Many ML projects are technically infeasible or poorly scoped.

- Many ML projects never make the leap into production.

- Many ML projects have unclear success criteria.

- Many ML projects are poorly managed.

## 	b. Lifecycle:

- Phase 1 is **Project Planning and Project Setup**: At this phase, we want to decide the problem to work on, determine the requirements and goals, as well as figure out how to allocate resources properly.

- Phase 2 is **Data Collection and Data Labeling**: At this phase, we want to collect training data (images, text, tabular, etc.) and potentially annotate them with ground truth, depending on the specific sources where they come from.

- Phase 3 is **Model Training and Model Debugging**: At this phase, we want to implement baseline models quickly, find and reproduce state-of-the-art methods for the problem domain, debug our implementation, and improve the model performance for specific tasks.

- Phase 4 is **Model Deployment and Model Testing**: At this phase, we want to pilot the model in a constrained environment, write tests to prevent regressions, and roll the model into production.

## 	c. Prioritizing:

- The project should have **high impact,** where cheap prediction is valuable for the complex parts of your business process.

- The project should have **high feasibility,** which is driven by the data availability, accuracy requirements, and problem difficulty.

- Here are 3 types of hard machine learning problems:
  - (1) The output is complex.
  - (2) Reliability is required.
  - (3) Generalization is expected.

## 	d. Archetypes:

- Archetype 1 - Projects that **improve an existing process**: improving route optimization in a ride-sharing service, building a customized churn prevention model, building a better video game AI.

- Archetype 2 - Projects that **augment a manual process**: turning mockup designs into application UI, building a sentence auto-completion feature, helping a doctor to do his/her job more efficient.

- Archetype 3 - Projects that **automate a manual process**: developing autonomous vehicles, automating customer service, automating website design.

**Data flywheels** is a concept saying that more users lead to more data, more data lead to better models, and better models lead to more users.

## 	e. Metrics:

- In most real-world projects, you usually care about a lot of metrics. Because machine learning systems work best when optimizing a single number, you need to pick a formula for combining different metrics of interest.

- The first way is to do a simple **average** (or **weighted average**) of these metrics.

- The second way is to choose a metric as a **threshold** and evaluate at that threshold value. The thresholding metrics are up to your domain judgment, but you would probably want to choose ones that are least sensitive to model choice and are closest to desirable values.

- The third way is to use a more **complex / domain-specific** formula. A solid process to go about this direction is to first start enumerating all the project requirements, then evaluate the current performance of your model, then compare the current performance to the requirements, and finally revisit the metric as your numbers improve.

## 	f. Baselines:

- A **baseline** is a model that is both simple to set up and has a reasonable chance of providing decent results. It gives you a lower bound on expected model performance.

- Your choice of a simple baseline depends on the kind of data you are working with and the kind of task you are targeting.

- You can look for **external baselines** such as business and engineering requirements, as well as published results from academic papers that tackle your problem domain.

- You can also look for **internal baselines** using simple models and human performance.

- Broadly speaking, the higher the quality of your baselines are, the easier it is to label more data. More **specialized domains** require more **skilled labelers**, so you should find cases **where the model performs worse** and **concentrate the data collection effort there**.



# 2. Infrastructure and Tooling

## 	a. Overview

- Google's seminal paper "Machine Learning: The High-Interest Credit Card of Technical Debt" states that if we look at the whole **machine learning system**, the **actual** modeling code is **very small**. There are **a lot of other code** around it that **configure** the system, **extract** the data/features, **test** the model performance, **manage** processes/resources, and **serve/deploy** the model.
- The **data component:**
  - **Data Storage** - How to **store** the data?
  - **Data Workflows** - How to **process** the data?
  - **Data Labeling** - How to **label** the data?
  - **Data Versioning** - How to **version** the data?
- The **development component:**
  - **Software Engineering** - How to choose the proper **engineering tools**?
  - **Frameworks** - How to choose the right **deep learning frameworks**?
  - **Distributed Training** - How to **train** the models in a **distributed fashion**?
  - **Resource Management** - How to **provision** and **mange** distributed **GPUs**?
  - **Experiment Management** - How to **manage** and **store** model **experiments**?
  - **Hyper-parameter Tuning** - How to **tune** model **hyper-parameters**?
- The **deployment component**
  - **Continuous Integration and Testing** - How to not break things as models are updated?
  - **Web** - How to deploy models to web services?
  - **Hardware and Mobile** - How to deploy models to embedded and mobile systems?
  - **Interchange** - How to deploy models across systems?
  - **Monitoring** - How to monitor model predictions?
- **All-In-One**: There are solutions that handle all of these components!

## 	b. Software Engineering

- **Python** is the clear programming language of choice.
- **Visual Studio Code** makes for a very nice Python experience, with features such as built-in git staging and diffing, peek documentation, and linter code hints.
- **PyCharm** is a popular choice for Python developers.
- **Jupyter Notebooks** is the standard tool for quick prototyping and exploratory analysis, but it is not suitable to build machine learning products.
- **Streamlit** is a new tool that fulfills a common need - an interactive applet to communicate the modeling results.

## 	c. Computing and GPUs

- If you go with the GPU round, there are a lot of **NVIDIA** cards to choose from (Kepler, Maxwell, Pascal, Volta, Turing).
- If you go with a cloud provider, **Amazon Web Services** and **Google Cloud Platform** are the heavyweights, while startups such as **Paperspace** and **Lambda Labs** are also viable options.
- If you work solo or in a startup, you should build or buy a 4x recent-architecture PC for model development. For model training, if you run many experiments, you can either buy shared server machines or use cloud instances.
- If you work in a large company, you are more likely to rely on cloud instances for both model development and model training, as they provide proper provisioning and infrastructure to handle failures.

## 	d. Resource Management

- Running complex deep learning models poses a very practical resource management problem: how to give every team the tools they need to train their models without requiring them to operate their own infrastructure?
- The most primitive approach is to use **spreadsheets** that allow people to reserve what resources they need to use.
- The next approach is to utilize a **SLURM Workload Manager**, a free and open-source job scheduler for Linux and Unix-like kernels.
- A very standard approach these days is to use Docker alongside Kubernetes.
  - **Docker** is a way to package up an entire dependency stack in a lighter-than-a-Virtual-Machine package.
  - **Kubernetes** is a way to run many Docker containers on top of a cluster.
- The last option is to use open-source projects.
  - Using **Kubeflow** allows you to run model training jobs at scale on containers with the same scalability of container orchestration that comes with Kubernetes.
  - **Polyaxon** is a self-service multi-user system, taking care of scheduling and managing jobs in order to make the best use of available cluster resources.

## 	e. Frameworks and Distributed Training

- Unless you have a good reason not to, you should use either **TensorFlow** or **PyTorch**.
- Both frameworks are converging to a point where they are good for research and production.
- [**fast.ai**](http://fast.ai/) is a solid option for beginners who want to iterate quickly.
- Distributed training of neural networks can be approached in 2 ways: (1) data parallelism and (2) model parallelism.
- Practically, **data parallelism** is more popular and frequently employed in large organizations for executing production-level deep learning algorithms.
- **Model parallelism**, on the other hand, is only necessary when a model does not fit on a single GPU.
- [**Ray**](http://docs.ray.io/) is an open-source project for effortless, stateful, parallel, and distributed computing in Python.
- [**RaySGD**](https://docs.ray.io/en/latest/raysgd/raysgd_pytorch.html) is a library for distributed data parallel training that provides fault tolerance and seamless parallelization, built on top of [**Ray**](http://docs.ray.io/).
- **Horovod** is Uber’s open-source distributed deep learning framework that uses a standard multi-process communication framework, so it can be an easier experience for multi-node training.

## 	f. Experiment Management

- Getting your models to perform well is a very iterative process. If you don’t have a system for managing your experiments, it quickly gets out of control.
- **TensorBoard** is a TensorFlow extension that allows you to easily monitor your model in a browser.
- **Losswise** provides ML practitioners with a Python API and accompanying dashboard to visualize progress within and across training sessions.
- [**Comet.ml**](http://comet.ml/) is another platform that enables engineers and data scientists to efficiently maintain their preferred workflow and tools, track previous work, and collaborate throughout the iterative process.
- **Weights & Biases** is an experiment tracking tool for deep learning that allows you to (1) store all the hyper-parameters and output metrics in one place; (2) explore and compare every experiment with training/inference visualizations; and (3) create beautiful reports that showcase your work.
- **MLflow** is an open-source platform for the entire machine learning lifecycle started by Databricks. Its MLflow Tracking component is an API and UI for logging parameters, code versions, metrics, and output files from your model training process.

## 	g. Hyperparameter Tuning

- Deep learning models are literally full of hyper-parameters. Finding the best configuration for these variables in a high-dimensional space is not trivial.
- Searching for hyper-parameters is an iterative process constrained by computing power, money, and time. Therefore, it would be really useful to have software that helps you search over hyper-parameter settings.
- **Hyperopt** is a Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.
- **SigOpt** is an optimization-as-a-service API that allows users to seamlessly tune the configuration parameters in AI and ML models.
- [**Ray Tune**](https://docs.ray.io/en/latest/tune.html) is a Python library for hyperparameter tuning at any scale, integrating seamlessly with optimization libraries such as **Hyperopt** and **SigOpt**.
- **Weights & Biases** has a nice feature called “Hyperparameter Sweeps” — a way to efficiently select the right model for a given dataset using the tool.

## 	h. All-in-one Solutions

- The “All-In-One” machine learning platforms provide a single system for everything: developing models, scaling experiments to many machines, tracking experiments and versioning models, deploying models, and monitoring model performance.
- **FBLearner Flow** is the workflow management platform at the heart of the Facebook ML engineering ecosystem.
- **Michelangelo**, Uber’s ML Platform, supports the training and serving of thousands of models in production across the company.
- **TensorFlow Extended** (TFX) is a Google-production-scale ML platform based on TensorFlow.
- Another option from Google is its **Cloud AI Platform**, a managed service that enables you to easily build machine learning models, that work on any type of data, of any size.
- **Amazon SageMaker** is one of the core AI offerings from AWS that helps teams through all stages in the machine learning life cycle.
- **Neptune** is a product that focuses on managing the experimentation process while remaining lightweight and easy to use by any data science team.
- **FloydHub** is another managed cloud platform for data scientists.
- **Paperspace** provides a solution for accessing computing power via the cloud and offers it through an easy-to-use console where everyday consumers can just click a button to log into their upgraded, more powerful remote machine.
- **Determined AI** is a startup that creates software to handle everything from managing cluster compute resources to automating workflows, thereby putting some of that big-company technology within reach of any organization.
- **Domino Data Lab** is an integrated end-to-end platform that is language agnostic, having a rich functionality for version control and collaboration; as well as one-click infrastructure scalability, deployment, and publishing.

# 3. Data Management

## 	a. Overview

- Data science has never been as much about machine learning as it has about cleaning, shaping, and moving data from place to place.
- Here are the important concepts in data management:
  - **Sources -** how to get training data
  - **Labeling -** how to label proprietary data at scale
  - **Storage -** how to store data and metadata appropriately
  - **Versioning -** how to update data through user activity or additional labeling
  - **Processing -** how to aggregate and convert raw data and metadata

## 	b. Sources

- Most deep learning applications require lots of labeled data. There are publicly available datasets that can serve as a starting point, but there is no competitive advantage of doing so.
- Most companies usually spend a lot of money and time to label their own data.
- **Data flywheel** means harnessing the power of users rapidly improve the whole machine learning system.
- **Semi-supervised learning** is a relatively recent learning technique where the training data is autonomously (or automatically) labeled.
- **Data augmentation** is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data.
- **Synthetic data** is data that’s generated programmatically, an underrated idea that is almost always worth starting with.

## 	c. Labeling

- Data labeling requires a collection of data points such as images, text, or audio and a qualified team of people to label each of the input points with meaningful information that will be used to train a machine learning model.
- You can create a **user interface** with a standard set of features (bounding boxes, segmentation, key points, cuboids, set of applicable classes…) and train your own annotators to label the data.
- You can leverage other labor sources by either **hiring** your own annotators or **crowdsourcing** the annotators.
- You can also consult standalone **service companies**. Data labeling requires separate software stack, temporary labor, and quality assurance; so it makes sense to **outsource**.

## 	d. Storage

- Data storage requirements for AI vary widely according to the application and the source material.
- The **filesystem** is the foundational layer of storage. Its fundamental unit is a “file” — which can be text or binary, is not versioned, and is easily overwritten.
- **Object storage** is an API over the filesystem that allows users to use a command on files (GET, PUT, DELETE) to a service, without worrying where they are actually stored. Its fundamental unit is an “object” — which is usually binary (images, sound files…).
- The **database** is a persistent, fast, and scalable storage/retrieval of structured data. Its fundamental unit is a “row” (unique IDs, references to other rows, values in columns).
- A **data lake** is the unstructured aggregation of data from multiple sources (databases, logs, expensive data transformations). It operates under the concept of “schema-on-read” by dumping everything in and then transforming the data for specific needs later.

## 	e. Versioning

- Data versioning refers to saving new copies of your data when you make changes so that you can go back and retrieve specific versions of your files later.
- In **Level 0**, the data lives on the filesystem and/or object storage and the database without being versioned.
- In **Level 1**, the data is versioned by storing a snapshot of everything at training time.
- In **Level 2**, the data is versioned as a mix of assets and code.
- **Level 3** requires specialized solutions for versioning data. You should avoid these until you can fully explain how they will improve your project.

## 	f. Processing

- The simplest thing we can do is a **Makefile** to specify what action(s) depend on.
- You will probably need a workflow management system. **Airflow** is the current winner of this space.
- Try to keep things simple and don't over-engineer your processing pipeline.

# 4. Machine Learning Teams

## 	a. Overview

- Machine Learning talents are expensive and scarce.
- Machine Learning teams have a diverse set of roles.
- Machine Learning projects have unclear timelines and high uncertainty.
- Machine Learning is also the “[high-interest credit card of technical debt](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)."
- Leadership often doesn’t understand Machine Learning.

## 	b. Roles

- The **Machine Learning Product Manager** is someone who works with the Machine Learning team, as well as other business functions and the end-users.
  - This person designs docs, creates wireframes, comes up with the plan to prioritize and execute Machine Learning projects.
  - The role is just like a traditional Product Manager, but with a deep knowledge of the Machine Learning development process and mindset.
- The **DevOps Engineer** is someone who deploys and monitors production systems.
  - This person handles the infrastructure that runs the deployed Machine Learning product.
  - This role is primarily a software engineering role, which often comes from a standard software engineering pipeline.
- The **Data Engineer** is someone who builds data pipelines, aggregates and collects from data storage, monitors data behavior.
  - This person works with distributed systems such as Hadoop, Kafka, Airflow.
  - This person belongs to the software engineering team that works actively with Machine Learning teams.
- The **Machine Learning Engineer** is someone who trains and deploys prediction models.
  - This person uses tools like TensorFlow and Docker to work with prediction systems running on real data in production.
  - This person is either an engineer with significant self-teaching OR a science/engineering Ph.D. who works as a traditional software engineer after graduate school.
- The **Machine Learning Researcher** is someone who trains prediction models, but often forward-looking or not production-critical.
  - This person uses TensorFlow / PyTorch / Jupiter to build models and reports describing their experiments.
  - This person is a Machine Learning expert who usually has an MS or Ph.D. degree in Computer Science or Statistics or finishes an industrial fellowship program.
- The **Data Scientist** is actually a blanket term used to describe all of the roles above.
  - In some organizations, this role actually entails answering business questions via analytics.
  - The role constitutes a wide range of backgrounds from undergraduate to Ph.D. students.

## 	c. Team Structure

- The **Nascent and Ad-Hoc Machine Learning** organization:
  - No one is doing Machine Learning, or Machine Learning is done on an ad-hoc basis.
  - There is often low-hanging fruit for Machine Learning.
  - But there is little support for Machine Learning projects and it’s very difficult to hire and retain good talent.
- The **Research and Development Machine Learning** organization:
  - Machine Learning efforts are centered in the R&D arm of the organization. Often hire Machine Learning researchers and doctorate students with experience publishing papers.
  - They can hire experienced researchers and work on long-term business priorities to get big wins.
  - However, it is very difficult to get quality data. Most often, this type of research work rarely translates into actual business value, so usually the amount of investment remains small.
- The **Business and Product Embedded Machine Learning** organization:
  - Certain product teams or business units have Machine Learning expertise along-side their software or analytics talent. These Machine Learning individuals report up to the team’s engineering/tech lead.
  - Machine Learning improvements are likely to lead to business value. Furthermore, there is a tight feedback cycle between idea iteration and product improvement.
  - Unfortunately, it is still very hard to hire and develop top talent, and access to data & compute resources can lag. There are also potential conflicts between Machine Learning project cycles and engineering management, so long-term Machine Learning projects can be hard to justify.
- The **Independent Machine Learning** organization:
  - Machine Learning division reports directly to senior leadership. The Machine Learning Product Managers work with Researchers and Engineers to build Machine Learning into client-facing products. They can sometimes publish long-term research.
  - Talent density allows them to hire and train top practitioners. Senior leaders can marshal data and compute resources. This gives the organizations to invest in tooling, practices, and culture around Machine Learning development.
  - A disadvantage is that model handoffs to different business lines can be challenging, since users need to buy-in to Machine Learning benefits and get educated on the model use. Also, feedback cycles can be slow.
- The **Machine Learning First** organization:
  - CEO invests in Machine Learning and there are experts across the business focusing on quick wins. The Machine Learning division works on challenging and long-term projects.
  - They have the best data access (data thinking permeates the organization), the most attractive recruiting funnel (challenging Machine Learning problems tends to attract top talent), and the easiest deployment procedure (product teams understand Machine Learning well enough).
  - This type of organization archetype is hard to implement in practice since it is culturally difficult to embed Machine Learning thinking everywhere.
- Organizational design follow 3 broad categories:
  - **Software Engineer vs Research**: To what extent is the Machine Learning team responsible for building or integrating with software? How important are Software Engineering skills on the team?
  - **Data Ownership**: How much control does the Machine Learning team have over data collection, warehousing, labeling, and pipelining?
  - **Model Ownership**: Is the Machine Learning team responsible for deploying models into production? Who maintains the deployed models?

## 	d. Managing Projects

- Manage Machine Learning projects can be **very challenging**:
  - In Machine Learning, it is hard to tell in advance what’s hard and what’s easy.
  - Machine Learning progress is nonlinear.
  - There are cultural gaps between research and engineering because of different values, backgrounds, goals, and norms.
  - Often, leadership just does not understand it.
- The secret sauce is to plan the Machine Learning project **probabilistically**!
  - Attempt a portfolio of approaches.
  - Measure progress based on inputs, not results.
  - Have researchers and engineers work together.
  - Get end-to-end pipelines together quickly to demonstrate quick wins.
  - Educate leadership on Machine Learning timeline uncertainty.

## 	e. Hiring

- Machine Learning talent is scarce.
- As a manager, be specific about what skills are must-have in the Machine Learning job descriptions.
- As a job seeker, it can be brutally challenging to break in as an outsider, so use projects as a signal to build awareness.

# 5. Training and Debugging

## 	a. Overview

- A common sentiment among practitioners is that they spend 80–90% of time debugging and tuning the models, and only 10–20% of time deriving math equations and implementing things.
- Reproducing the results in deep learning can be challenging due to various factors including implementation bugs, choices of model hyper-parameters, data/model fit, and the construction of data.

## 	b. Start Simple

- **Choose simple architecture**:
  - LeNet/ResNet for images.
  - LSTM for sequences.
  - Fully-connected network with one hidden layer for all other tasks.
- **Use sensible hyper-parameter defaults**:
  - Adam optimizer with a “magic” learning rate value of 3e-4.
  - ReLU activation for fully-connected and convolutional models and TanH activation for LSTM models.
  - He initialization for ReLU and Glorot initialization for TanH.
  - No regularization and data normalization.
- **Normalize data inputs**: subtracting the mean and dividing by the variance.
- **Simplify the problem**:
  - Working with a small training set around 10,000 examples.
  - Using a fixed number of objects, classes, input size, etc.
  - Creating a simpler synthetic training set like in research labs.

## 	c. Debug

- The 5 most common bugs in deep learning models include:
  - Incorrect shapes for tensors.
  - Pre-processing inputs incorrectly.
  - Incorrect input to the loss function.
  - Forgot to set up train mode for the network correctly.
  - Numerical instability - inf/NaN.
- 3 pieces of general advice for implementing models:
  - Start with **a lightweight implementation**.
  - Use **off-the-shelf components** such as Keras if possible, since most of the stuff in Keras works well out-of-the-box.
  - Build **complicated data pipelines later**.
- The first step is to **get the model to run**:
  - For **shape mismatch and casting issues**, you should step through your model creation and inference step-by-step in a debugger, checking for correct shapes and data types of your tensors.
  - For **out-of-memory issues**, you can scale back your memory-intensive operations one-by-one.
  - For **other issues**, simply Google it. StackOverflow would be great most of the time.
- The second step is to have the model **overfit a single batch**:
  - **Error goes up:** Commonly this is due to a flip sign somewhere in the loss function/gradient.
  - **Error explodes:** This is usually a numerical issue, but can also be caused by a high learning rate.
  - **Error oscillates:** You can lower the learning rate and inspect the data for shuffled labels or incorrect data augmentation.
  - **Error plateaus:** You can increase the learning rate and get rid of regulation. Then you can inspect the loss function and the data pipeline for correctness.
- The third step is to **compare the model to a known result**:
  - The most useful results come from **an official model implementation** **evaluated on a similar dataset to yours**.
  - If you can’t find an official implementation on a similar dataset, you can compare your approach to results from **an official model implementation evaluated on a benchmark dataset**.
  - If there is no official implementation of your approach, you can compare it to results from **an unofficial model implementation**.
  - Then, you can compare to results from **a paper with no code**, results from **the model on a benchmark dataset**, and results from **a similar model on a similar dataset**.
  - An under-rated source of results come from **simple baselines**, which can help make sure that your model is learning anything at all.

## 	d. Evaluate

- You want to apply **the bias-variance decomposition** concept here: *Test error = irreducible error + bias + variance + validation overfitting*.
- If the training, validation, and test sets come from different data distributions, then you should use **2 validation sets**: one set sampled from the training distribution, and the other set sampled from the test distribution.

## 	e. Improve

- The first step is to **address under-fitting:**
  - Add model complexity → Reduce regularization → Error analysis → Choose a more complex architecture → Tune hyper-parameters → Add features.
- The second step is to **address over-fitting:**
  - Add more training data → Add normalization → Add data augmentation → Increase regularization → Error analysis → Choose a more complex architecture → Tune hyper-parameters → Early stopping → Remove features → Reduce model size.
- The third step is to **address the distribution shift** present in the data:
  - Analyze test-validation errors and collect more training data to compensate.
  - Analyze test-validation errors and synthesize more training data to compensate.
  - Apply domain adaptation techniques to training and test distributions.
- The final step, if applicable, is to **rebalance your datasets:**
  - If the model performance on the test & validation set is significantly better than the performance on the test set, you over-fit to the validation set.
  - When it does happen, you can recollect the validation data by re-shuffling the test/validation split ratio.

## 	f. Tune

- Choosing which hyper-parameters to optimize is not an easy task since some are more sensitive than others and are dependent upon the choice of model.
  - **Low sensitivity**: Optimizer, batch size, non-linearity.
  - **Medium sensitivity**: weight initialization, model depth, layer parameters, weight of regularization.
  - **High sensitivity**: learning rate, annealing schedule, loss function, layer size.
- Method 1 is **manual optimization:**
  - For a skilled practitioner, this may require the least amount of computation to get good results.
  - However, the method is time-consuming and requires a detailed understanding of the algorithm.
- Method 2 is **grid search:**
  - Grid search is super simple to implement and can produce good results.
  - Unfortunately, it’s not very efficient since we need to train the model on all cross-combinations of the hyper-parameters. It also requires prior knowledge about the parameters to get good results.
- Method 3 is **random search:**
  - Random search is also easy to implement and often produces better results than grid search.
  - But it is not very interpretable and may also require prior knowledge about the parameters to get good results.
- Method 4 is **coarse-to-fine search:**
  - This strategy helps you narrow in only on very high performing hyper-parameters and is a common practice in the industry.
  - The only drawback is that it is somewhat a manual process.
- Method 5 is **Bayesian optimization search:**
  - Bayesian optimization is generally the most efficient hands-off way to choose hyper-parameters.
  - But it’s difficult to implement from scratch and can be hard to integrate with off-the-shelf tools.

## 	g. Conclusion

- Deep learning debugging is hard due to many competing sources of error.
- To train bug-free deep learning models, you need to treat building them as an iterative process.
  - Choose the simplest model and data possible.
  - Once the model runs, overfit a single batch and reproduce a known result.
  - Apply the bias-variance decomposition to decide what to do next.
  - Use coarse-to-fine random searches to tune the model’s hyper-parameters.
  - Make your model bigger if your model under-fits and add more data and/or regularization if your model over-fits.

# 6. Testing and Deployment

## 	a. Project Structure

- The **prediction system** involves code to process input data, to construct networks with trained weights, and to make predictions.
- The **training system** processes raw data, runs experiments, and manages results.
- The goal of any prediction system is to be deployed into the **serving system**. Its purpose is to serve predictions and to scale to demand.
- **Training and validation data** are used in conjunction with the training system to generate the prediction system.
- At production time, we have **production data** that has not been seen before and can only be served by the serving system.
- The prediction system should be tested by **functionality** to catch code regressions and by **validation** to catch model regressions.
- The training system should have its tests to catch upstream regressions (change in data sources, upgrade of dependencies)
- For production data, we need **monitoring** that raises alert to downtime, errors, distribution shifts, etc. and catches service and data regressions.

## 	b. ML Test Score

- [ML Test Score :  A Rubric for Production Readiness and Technical Debt Reduction](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)  is an exhaustive framework/checklist from practitioners at Google.
- The paper presents a rubric as a set of 28 actionable tests and offers a scoring system to measure how ready for production a given machine learning system is. These are categorized into 4 sections: (1) data tests, (2) model tests, (3) ML infrastructure tests, and (4) monitoring tests.
- The scoring system provides a vector for incentivizing ML system developers to achieve stable levels of reliability by providing a clear indicator of readiness and clear guidelines for how to improve.

## 	c. CI / Testing

- **Unit tests** are designed for specific module functionality.
- **Integration tests** are designed for the whole system.
- **Continuous integration** is an environment where tests are run every time a new code is pushed to the repository before the updated model is deployed.
- A quick survey of continuous integration tools yields several options: CircleCI, Travis CI, Jenkins, and Buildkite.

## 	d. Docker

- Docker is a computer program that performs operating-system-level virtualization, also known as **containerization**.
- A container is a standardized unit of fully packaged software used for local development, shipping code, and deploying system.
- Though Docker presents on how to deal with each of the individual microservices, we also need an orchestrator to handle the whole cluster of services. For that, **Kubernetes** is the open-source winner, and it has excellent support from the leading cloud vendors.

## 	e. Web Deployment

- For web deployment, you need to be familiar with the concept of **REST API.**
  - You can deploy the code to Virtual Machines, and then scale by adding instances.
  - You can deploy the code as containers, and then scale via orchestration.
  - You can deploy the code as a “server-less function.”
  - You can deploy the code via a model serving solution.
- If you are making **CPU inference**, you can get away with scaling by launching more servers (Docker), or going serverless (AWS Lambda).
- If you are using **GPU inference**, things like TF Serving and Clipper become useful with features such as adaptive batching.

## 	f. Monitoring

- It is crucial to monitor serving systems, training pipelines, and input data. A typical monitoring system can **raise alarms** when things go wrong and provide the records for tuning things.
- Cloud providers have decent monitoring solutions.
- Anything that can be logged can be monitored: dependency changes, distribution shift in data, model instabilities, etc.
- **Data distribution monitoring** is an underserved need!
- It is important to monitor the **business uses** of the model, not just its statistics. Furthermore, it is important to be able to **contribute failures** back to the dataset.

## 	g. Hardware/Mobile

- Embedded and mobile devices have low-processor with little memory, which makes the process slow and expensive to compute. Often, we can try some tricks such as reducing network size, quantizing the weights, and distilling knowledge.
  - Both **pruning** and **quantization** are model compression techniques that make the model physically smaller to save disk space and make the model require less memory during computation to run faster.
  - **Knowledge distillation** is a compression technique in which a small “student” model is trained to reproduce the behavior of a large “teacher” model.
- Embedded and mobile PyTorch/TensorFlow frameworks are less fully featured than the full PyTorch/TensorFlow frameworks. Therefore, we have to be careful with the model architecture. An alternative option is using the interchange format.
  - **Mobile machine learning frameworks** are regularly in flux: Tensorflow Lite, PyTorch Mobile, CoreML, MLKit, FritzAI.
  - The best solution in the industry for **embedded** devices is NVIDIA.
  - The **Open Neural Network Exchange** (ONNX for short) is designed to allow framework interoperability.

# 7. Research Areas

# 8. Labs

# 9. Where to go next