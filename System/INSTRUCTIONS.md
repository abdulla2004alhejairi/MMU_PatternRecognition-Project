## How To Run
1. Install `virtualenv`:
```
$ pip install virtualenv
```

2. Open a terminal in the project root directory and run:
```
$ virtualenv env
```

3. Then run the command:
```
$ .\env\Scripts\activate
```

4. Then install the dependencies:
```
$ (env) pip install -r requirements.txt
```

5. Optional: Download and extract the 'obj' and 'algo' image folder to this directory:
...
https://mmuedumy-my.sharepoint.com/:f:/g/personal/1211105012_student_mmu_edu_my/EiWD5xanjqJMomoHL-MSlJoBuuPtDdu92qDjpVH9zucqUA?e=X2MAO2
...

6. Train and export model (Only if 5 is complete):
```
$ (env) python model.py
```

7. Finally start the web server:
```
$ (env) python app.py
```
