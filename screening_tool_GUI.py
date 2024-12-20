# Import necessary libraries
import os
import time
import utils
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import numpy as np
import tkinter as tk
from tkinter import ttk
from termcolor import colored
import threading

# Function definitions
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def string_call(c):
    return f"{c['Title']} - {c['URL']}"

def string_site(s):
    return f"{s['scraped'][0][1]['Title']} - {s['scraped'][0][1]['URL']}"

def calc_sim(eu, site):
    return round(np.float64(cos_sim(eu["Title_Embedding"], site["Summary_Embedding"])), 3) * 100

def progress_bar(x, n=100):
    progress = int(n * x)
    return "[" + ("█" * progress) + (" " * (n-progress)) + "]"

# Load data and model outside of GUI functions to avoid reloading
print("Loading data and model. Please wait...")
with open('all_data_embedded.json', 'r', encoding='utf-8') as f:
    (calls, sites) = json.load(f)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
comps = {c: model.encode(c) for c in utils.competencies}

# Calculate similarity matrix between calls and sites
sim_matrix = [[calc_sim(call, site) for call in calls] for site in sites]
sims = np.array(sim_matrix)

# -------------------------------
# GUI Application
# -------------------------------

def main():
    # Create the main window
    root = tk.Tk()
    root.title("TI EU Funding Helper")
    root.geometry("800x600")

    # Create tabs
    tab_control = ttk.Notebook(root)

    tab_similarity = ttk.Frame(tab_control)
    tab_funding_calls = ttk.Frame(tab_control)
    tab_company_search = ttk.Frame(tab_control)
    tab_competencies = ttk.Frame(tab_control)
    tab_matches = ttk.Frame(tab_control)
    tab_match_details = ttk.Frame(tab_control)

    tab_control.add(tab_similarity, text='Text Similarity')
    tab_control.add(tab_funding_calls, text='Funding Calls')
    tab_control.add(tab_company_search, text='Company Search')
    tab_control.add(tab_competencies, text='Competencies')
    tab_control.add(tab_matches, text='Top Matches')
    tab_control.add(tab_match_details, text='Match Details')

    tab_control.pack(expand=1, fill='both')

    # Text Similarity Tab
    def calculate_similarity():
        text1 = entry_text1.get()
        text2 = entry_text2.get()
        if text1 and text2:
            sim = round(cos_sim(model.encode(text1), model.encode(text2)) * 100, 1)
            result = f"'{text1}' and '{text2}' are {sim}% similar"
            txt_similarity_output.config(state='normal')
            txt_similarity_output.delete(1.0, tk.END)
            txt_similarity_output.insert(tk.END, result)
            txt_similarity_output.config(state='disabled')
        else:
            tk.messagebox.showwarning("Input Error", "Please enter both texts.")

    lbl_text1 = tk.Label(tab_similarity, text="Enter first text:")
    lbl_text1.pack()
    entry_text1 = tk.Entry(tab_similarity, width=80)
    entry_text1.pack()
    lbl_text2 = tk.Label(tab_similarity, text="Enter second text:")
    lbl_text2.pack()
    entry_text2 = tk.Entry(tab_similarity, width=80)
    entry_text2.pack()
    btn_similarity = tk.Button(tab_similarity, text="Calculate Similarity", command=calculate_similarity)
    btn_similarity.pack()
    txt_similarity_output = tk.Text(tab_similarity, height=5, state='disabled')
    txt_similarity_output.pack()

    # Funding Calls Tab
    def search_funding_calls():
        query = entry_funding_query.get()
        try:
            num_results = int(entry_funding_results.get())
        except ValueError:
            num_results = 10
        output = ""
        if query.strip():
            q = model.encode(query)
            results = [{"score": cos_sim(q, c['Title_Embedding']) * 100,
                        "call": c,
                        "index": i}
                       for i, c in enumerate(calls)]
            for r in sorted(results, key=lambda x: x["score"], reverse=True)[:min(num_results, len(calls))]:
                output += f"{round(r['score'], 1)}%\n"
                output += f"#{r['index']} {string_call(r['call'])}\n\n"
        else:
            for i, c in enumerate(calls[:min(num_results, len(calls))]):
                output += f"#{i} {string_call(c)}\n\n"
        txt_funding_output.config(state='normal')
        txt_funding_output.delete(1.0, tk.END)
        txt_funding_output.insert(tk.END, output)
        txt_funding_output.config(state='disabled')

    lbl_funding_query = tk.Label(tab_funding_calls, text="Search query (leave empty to list all):")
    lbl_funding_query.pack()
    entry_funding_query = tk.Entry(tab_funding_calls, width=80)
    entry_funding_query.pack()
    lbl_funding_results = tk.Label(tab_funding_calls, text="Number of results:")
    lbl_funding_results.pack()
    entry_funding_results = tk.Entry(tab_funding_calls, width=10)
    entry_funding_results.insert(0, "10")
    entry_funding_results.pack()
    btn_funding_search = tk.Button(tab_funding_calls, text="Search Funding Calls", command=search_funding_calls)
    btn_funding_search.pack()
    txt_funding_output = tk.Text(tab_funding_calls, state='disabled')
    txt_funding_output.pack(expand=1, fill='both')

    # Company Search Tab
    def search_companies():
        query = entry_company_query.get()
        try:
            num_results = int(entry_company_results.get())
        except ValueError:
            num_results = 10
        output = ""
        if query.strip():
            q = model.encode(query)
            results = [{"score": cos_sim(q, s['Summary_Embedding']) * 100,
                        "site": s,
                        "index": i}
                       for i, s in enumerate(sites)]
            for r in sorted(results, key=lambda x: x["score"], reverse=True)[:min(num_results, len(sites))]:
                output += f"{round(r['score'], 1)}%\n"
                output += f"#{r['index']} {string_site(r['site'])}\n\n"
        else:
            for i, s in enumerate(sites[:min(num_results, len(sites))]):
                output += f"#{i} {string_site(s)}\n\n"
        txt_company_output.config(state='normal')
        txt_company_output.delete(1.0, tk.END)
        txt_company_output.insert(tk.END, output)
        txt_company_output.config(state='disabled')

    lbl_company_query = tk.Label(tab_company_search, text="Search query (leave empty to list all):")
    lbl_company_query.pack()
    entry_company_query = tk.Entry(tab_company_search, width=80)
    entry_company_query.pack()
    lbl_company_results = tk.Label(tab_company_search, text="Number of results:")
    lbl_company_results.pack()
    entry_company_results = tk.Entry(tab_company_search, width=10)
    entry_company_results.insert(0, "10")
    entry_company_results.pack()
    btn_company_search = tk.Button(tab_company_search, text="Search Companies", command=search_companies)
    btn_company_search.pack()
    txt_company_output = tk.Text(tab_company_search, state='disabled')
    txt_company_output.pack(expand=1, fill='both')

    # Competencies Tab
    def show_competencies():
        try:
            call_number = int(entry_call_number.get())
            if not (0 <= call_number < len(calls)):
                tk.messagebox.showwarning("Input Error", f"Call number must be between 0 and {len(calls)-1}")
                return
        except ValueError:
            tk.messagebox.showwarning("Input Error", "Please enter a valid call number.")
            return
        try:
            num_competencies = int(entry_competencies_number.get())
        except ValueError:
            num_competencies = 10
        output = f"#{call_number} {string_call(calls[call_number])}\n\n"
        results = [{"score": cos_sim(calls[call_number]['Title_Embedding'], e) * 100,
                    "comp": c}
                   for c, e in comps.items()]
        for r in sorted(results, key=lambda x: x["score"], reverse=True)[:min(num_competencies, len(comps))]:
            bar = progress_bar(r["score"] / 100)
            output += f"{bar} {round(r['score'], 1)}%\n {r['comp']}\n\n"
        txt_competencies_output.config(state='normal')
        txt_competencies_output.delete(1.0, tk.END)
        txt_competencies_output.insert(tk.END, output)
        txt_competencies_output.config(state='disabled')

    lbl_call_number = tk.Label(tab_competencies, text="Enter call number (#):")
    lbl_call_number.pack()
    entry_call_number = tk.Entry(tab_competencies, width=10)
    entry_call_number.pack()
    lbl_competencies_number = tk.Label(tab_competencies, text="Number of competencies:")
    lbl_competencies_number.pack()
    entry_competencies_number = tk.Entry(tab_competencies, width=10)
    entry_competencies_number.insert(0, "10")
    entry_competencies_number.pack()
    btn_show_competencies = tk.Button(tab_competencies, text="Show Competencies", command=show_competencies)
    btn_show_competencies.pack()
    txt_competencies_output = tk.Text(tab_competencies, state='disabled')
    txt_competencies_output.pack(expand=1, fill='both')

    # Top Matches Tab
    def show_top_matches():
        def process():
            choice = combo_match_choice.get()
            try:
                num_results = int(entry_matches_results.get())
            except ValueError:
                num_results = 10
            output = ""
            if choice == 'Specific call':
                try:
                    call_number = int(entry_match_call_number.get())
                    if not (0 <= call_number < len(calls)):
                        tk.messagebox.showwarning("Input Error", f"Call number must be between 0 and {len(calls)-1}")
                        return
                except ValueError:
                    tk.messagebox.showwarning("Input Error", "Please enter a valid call number.")
                    return
                site_sims = sims[:, call_number]
                top_matches_col = np.argsort(site_sims)[-num_results:]
                output += f"\n#{call_number} {string_call(calls[call_number])}\nMATCHES WITH\n\n"
                for row_index in reversed(top_matches_col):
                    value = round(site_sims[row_index], 1)
                    output += f"{value}%\n"
                    output += f"#{row_index} {string_site(sites[row_index])}\n\n"
            elif choice == 'Specific site':
                try:
                    site_number = int(entry_match_site_number.get())
                    if not (0 <= site_number < len(sites)):
                        tk.messagebox.showwarning("Input Error", f"Site number must be between 0 and {len(sites)-1}")
                        return
                except ValueError:
                    tk.messagebox.showwarning("Input Error", "Please enter a valid site number.")
                    return
                call_sims = sims[site_number]
                top_matches_row = np.argsort(call_sims)[-num_results:]
                output += f"\n#{site_number} {string_site(sites[site_number])}\nMATCHES WITH\n\n"
                for col_index in reversed(top_matches_row):
                    value = round(call_sims[col_index], 1)
                    output += f"{value}%\n"
                    output += f"#{col_index} {string_call(calls[col_index])}\n\n"
            else:
                top_matches = np.unravel_index(np.argsort(sims, axis=None)[-num_results:], sims.shape)
                top_matches = tuple(reversed(list(zip(*top_matches))))
                for index in top_matches:
                    value = round(sims[index], 1)
                    output += f"#{index[1]} {string_call(calls[index[1]])}\n"
                    output += f"MATCHED WITH {value}% TO\n"
                    output += f"#{index[0]} {string_site(sites[index[0]])}\n\n"
            txt_matches_output.config(state='normal')
            txt_matches_output.delete(1.0, tk.END)
            txt_matches_output.insert(tk.END, output)
            txt_matches_output.config(state='disabled')

        threading.Thread(target=process).start()

    lbl_match_choice = tk.Label(tab_matches, text="Select match type:")
    lbl_match_choice.pack()
    combo_match_choice = ttk.Combobox(tab_matches, values=['Specific call', 'Specific site', 'All'], state='readonly')
    combo_match_choice.current(0)
    combo_match_choice.pack()
    lbl_match_call_number = tk.Label(tab_matches, text="Enter call number (#):")
    lbl_match_site_number = tk.Label(tab_matches, text="Enter site number (#):")
    entry_match_call_number = tk.Entry(tab_matches, width=10)
    entry_match_site_number = tk.Entry(tab_matches, width=10)
    def update_match_inputs(event):
        choice = combo_match_choice.get()
        if choice == 'Specific call':
            lbl_match_site_number.pack_forget()
            entry_match_site_number.pack_forget()
            lbl_match_call_number.pack()
            entry_match_call_number.pack()
        elif choice == 'Specific site':
            lbl_match_call_number.pack_forget()
            entry_match_call_number.pack_forget()
            lbl_match_site_number.pack()
            entry_match_site_number.pack()
        else:
            lbl_match_call_number.pack_forget()
            entry_match_call_number.pack_forget()
            lbl_match_site_number.pack_forget()
            entry_match_site_number.pack_forget()
    combo_match_choice.bind("<<ComboboxSelected>>", update_match_inputs)
    update_match_inputs(None)
    lbl_matches_results = tk.Label(tab_matches, text="Number of results:")
    lbl_matches_results.pack()
    entry_matches_results = tk.Entry(tab_matches, width=10)
    entry_matches_results.insert(0, "10")
    entry_matches_results.pack()
    btn_show_matches = tk.Button(tab_matches, text="Show Top Matches", command=show_top_matches)
    btn_show_matches.pack()
    txt_matches_output = tk.Text(tab_matches, state='disabled')
    txt_matches_output.pack(expand=1, fill='both')

    # Match Details Tab
    def show_match_details():
        def process():
            try:
                call_number = int(entry_detail_call_number.get())
                if not (0 <= call_number < len(calls)):
                    tk.messagebox.showwarning("Input Error", f"Call number must be between 0 and {len(calls)-1}")
                    return
            except ValueError:
                tk.messagebox.showwarning("Input Error", "Please enter a valid call number.")
                return
            try:
                site_number = int(entry_detail_site_number.get())
                if not (0 <= site_number < len(sites)):
                    tk.messagebox.showwarning("Input Error", f"Site number must be between 0 and {len(sites)-1}")
                    return
            except ValueError:
                tk.messagebox.showwarning("Input Error", "Please enter a valid site number.")
                return
            try:
                num_competencies = int(entry_detail_competencies_number.get())
            except ValueError:
                num_competencies = 10
            output = f"\n#{call_number} {string_call(calls[call_number])}\nMATCHED WITH {round(sims[site_number, call_number], 1)}% TO\n"
            output += f"#{site_number} {string_site(sites[site_number])}\n\n"
            results = [{"score": cos_sim(calls[call_number]['Title_Embedding'], e) * 100,
                        "comp": c,
                        "embedding": e}
                       for c, e in comps.items()]
            for r in sorted(results, key=lambda x: x["score"], reverse=True)[:num_competencies]:
                bar_call = progress_bar(r["score"] / 100)
                site_score = cos_sim(sites[site_number]['Summary_Embedding'], r["embedding"]) * 100
                bar_site = progress_bar(site_score / 100)
                output += f"Competency: {r['comp']}\n"
                output += f"Call: {bar_call} {round(r['score'], 1)}%\n"
                output += f"Site: {bar_site} {round(site_score, 1)}%\n\n"
            txt_match_details_output.config(state='normal')
            txt_match_details_output.delete(1.0, tk.END)
            txt_match_details_output.insert(tk.END, output)
            txt_match_details_output.config(state='disabled')

        threading.Thread(target=process).start()

    lbl_detail_call_number = tk.Label(tab_match_details, text="Enter call number (#):")
    lbl_detail_call_number.pack()
    entry_detail_call_number = tk.Entry(tab_match_details, width=10)
    entry_detail_call_number.pack()
    lbl_detail_site_number = tk.Label(tab_match_details, text="Enter site number (#):")
    lbl_detail_site_number.pack()
    entry_detail_site_number = tk.Entry(tab_match_details, width=10)
    entry_detail_site_number.pack()
    lbl_detail_competencies_number = tk.Label(tab_match_details, text="Number of competencies:")
    lbl_detail_competencies_number.pack()
    entry_detail_competencies_number = tk.Entry(tab_match_details, width=10)
    entry_detail_competencies_number.insert(0, "10")
    entry_detail_competencies_number.pack()
    btn_show_match_details = tk.Button(tab_match_details, text="Show Match Details", command=show_match_details)
    btn_show_match_details.pack()
    txt_match_details_output = tk.Text(tab_match_details, state='disabled')
    txt_match_details_output.pack(expand=1, fill='both')

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
