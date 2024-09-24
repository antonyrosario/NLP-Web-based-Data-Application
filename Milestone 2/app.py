import os
import pickle

from flask import Flask, render_template, request, redirect, session
from gensim.models.fasttext import FastText
from bs4 import BeautifulSoup

from utils import docvecs
app = Flask(__name__)
app.secret_key = os.urandom(16)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/Accounting_Finance')
def Accounting_Finance():
    return render_template('Accounting_Finance.html')

@app.route('/Engineering')
def Engineering():
    return render_template('Engineering.html')

@app.route('/Healthcare_Nursing')
def Healthcare_Nursing():
    return render_template('Healthcare_Nursing.html')

@app.route('/Sales')
def Sales():
    return render_template('Sales.html')

@app.route('/Contact_Us')
def Contact_Us():
    return render_template('Contact_Us.html')

@app.route('/<folder>/<filename>')
def article(folder, filename):
    return render_template('/' + folder + '/' + filename + '.html')


@app.route("/admin", methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        # Read the .txt file
        f_title = request.form['title']
        f_content = request.form['description']

        # Classify the content
        if request.form['button'] == 'Classify':

            # Tokenize the content of the .txt file so as to input to the saved model - can do more complicated things here but as an example in the exercise, we just stop here
            tokenized_data = f_content.split(' ')

            # Load the FastText model
            descFT = FastText.load("desc_FT.model")
            descFT_wv = descFT.wv

            # Generate vector representation of the tokenized data
            descFT_dvs = docvecs(descFT_wv, [tokenized_data])

            # Load the LR model
            pkl_filename = "descFT_LR.pkl"
            with open(pkl_filename, 'rb') as file:
                model = pickle.load(file)

            # Predict the label of tokenized_data
            y_pred = model.predict(descFT_dvs)
            y_pred = y_pred[0]

            return render_template('admin.html', prediction=y_pred, title=f_title, description=f_content)

        elif request.form['button'] == 'Save':

            # First check if the recommended category is empty
            cat_recommend = request.form['category']
            if cat_recommend == '':
                return render_template('admin.html', prediction=cat_recommend,
                                       title=f_title, description=f_content,
                                       category_flag='Recommended category must not be empty.')

            elif cat_recommend not in ['Accounting_Finance', 'Engineering', 'Healthcare_Nursing', 'Sales']:
                return render_template('admin.html', prediction=cat_recommend,
                                       title=f_title, description=f_content,
                                       category_flag='Recommended category must belong to: Accounting_Finance, Engineering, Healthcare_Nursing, Sales.')

            else:

                # First read the html template
                soup = BeautifulSoup(open('templates/article_template.html'), 'html.parser')

                # Then adding the title and the content to the template
                # First, add the title
                div_page_title = soup.find('div', {'class': 'title'})
                title = soup.new_tag('h1', id='data-title')
                title.append(f_title)
                div_page_title.append(title)

                # Second, add the content
                div_page_content = soup.find('div', {'class': 'data-article'})
                content = soup.new_tag('p')
                content.append(f_content)
                div_page_content.append(content)

                # Finally write to a new html file
                filename_list = f_title.split()
                filename = '_'.join(filename_list)
                filename = cat_recommend + '/' + filename + ".html"
                with open("templates/" + filename, "w", encoding='utf-8') as file:
                    print(filename)
                    file.write(str(soup))

                # Clear the add-new-entry form and ask if user wants to continue to add new entry
                return redirect('/' + filename.replace('.html', ''))
    else:
        return render_template('admin.html')



@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':

        if request.form['search'] == '':
            search_string = request.form["searchword"]

            # search over all the html files in templates to find the search_string
            article_search = []
            dir_path = 'templates'
            for folder in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, folder)):
                    for filename in sorted(os.listdir(os.path.join(dir_path, folder))):
                        if filename.endswith('html'):
                            with open(os.path.join(dir_path, folder, filename), encoding="utf8") as file:
                                file_content = file.read()

                                # search for the string within the file
                                if search_string in file_content:
                                    article_search.append([folder, filename.replace('.html', '')])

            # generate the right format for the Jquery script in search.html
            num_results = len(article_search)

            # exact search or related search (regex, stemming or lemmatizing)

            # can handle the case when no search results

            # search both titles and descriptions

            return render_template('search.html', num_results=num_results, search_string=search_string,
                                   article_search=article_search)

    else:
        return render_template('home.html')

