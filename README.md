# tensorframe

## Authors note
Initiated this project to opensource code to GPU accelerate large scale data processing, with a focus on timeseries and tensor data), to solve for the many ETL and data cleaning related problems out there. Written in Python with C(++) extensions and CUDA. 

### Background
Why start with a CSV parser? First. Its a relatively simple problem to solve. Well, in theory. Practice proofed otherwise. Second reason. Most data is still exchanged in (ancient) CSV format, while parsers top out at 100MB/s. Some do reach GB/s, but quite often at a functionality trade-off, and /or still requires reparsing to a format we can do data science on. To solve the performance part we use GPU. Meanwhile, by structuring it in the (awesome) Apache Arrow standard we can immediately apply other cool functions, making data science on large datasets alot easier.

As exciting as CSV files are. You might have noticed the name of the project suggests something else. In a nutshell this project is about accelerating data processing. Needless to say. Compute requirements grow bigger by the day as we are entering the AI era, while traditional (serial, CPU) processing has stalled. In the opinion of many, the future of computing is massively parallel mixed with lots of machine learning based code.

### Status

#### 20190115
Added buildscripts to create Dockerbased builds for user and developer mode. This should have been topic no. 0. Took more time than expected. Fortunately it has been worth it. Installing and testing on new systems is now fully automated. More importantly, having this containerized makes it highly portable to cloud stacks and new system builds.

#### 20181231
First part is published. Starting with an "alpha-build" of the CSV parser. Its already over a magnitude faster than traditional CPU counterparts (and a few times better than a current GPU algorithm). But do note, Current version should still be considered alpha-build status. 

Regarding works-in-progress. Main topics are: 1. improve error checking and validation, 2. create some iPython examples for easy use, 3. convert handwritten designs to something more formal, and finally 4. integrate benchmark function to proof this really is the fastest and best CSV parser available ;).

### Long term view
There are many related projects attacking a similar problem or two. In fact, this space is evolving extremely fast and highly dynamic. Making it impossible to provide a real long term roadmap. Therefore, focus of this project is not to re-invent the wheel on anything cutting edge, but merely to integrate and make such tools accesible within a data processing pipeline. That is also why I choose to use Apache Arrow. Expecting to pick the early fruits of that very soon. Either way. TO BE CONTINUED.

## How to use?
This is one of the works-in-progress (see status). If you are brave you could follow the test/test_full.py code. You also need an updated Linux/ CUDA setup (tip: nvidia-docker) and a modern GPU (should work on Pascal 1060+, Turing, Volta). If you dont have such a setup you can of course get a GPU instance at AWS. For the more patient, I will work out some easy-to-follow examples soon.
