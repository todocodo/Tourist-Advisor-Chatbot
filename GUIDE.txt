1. Open the git-bash terminal (this terminal is automatically installed after the installation of git)
	- try using git-bash and not other terminal as I tried with the default "cmd" for windows but certain commands didn't work for some reason

2. Usually for new flask app a virtual environment needs to be created, however in this case I already created one, so proceed to the next step

3. run "source virt/Scripts/activate" to activate the virtual environmnet --> it should display "(virt)" after each command in the git-bash

4. run "pip install -r requirements.txt" to install all the requirements needed for the chatbot - You will probably
		add some new ones for your models but the 4 main libraries are added - keras, tensorflow, pandas, pickle

	IMPORTANT - for the requirements.txt - you may need to go into the file and adjust the versions of keras and tensorflow to
		    match the versions on which you model was trained - this is really important

5. to see which libraries are installed use - "pip freeze" - it should be used after the installation of requirements.txt 

6. run "export FLASK_ENV=development" to set the environment to development 

7. run "export FLASK_APP=main.py" to set a file from which to run the flask 

8. Finally run the server - "flask run" 


FOR THE MODEL INTEGRATION:

only the main.py is needed 

1. import all of your libraries
2. load the model and utilis using pickle or sth else
	include this above the "@app.route("/")
3. make sure the versions of the libraries in flask are the same as the ones you used to train your model
4. In the results() method add your logic for prediction, (preprocessing the text, or whatever you have), use my code for reference
5. Use the same method to display some results in the templates