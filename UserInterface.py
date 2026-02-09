from tkinter import *
import subprocess
import threading

'''
AI Disclaimer: Parts of this code were generated using AI.

To correctly run this file, ensure that transcribe2.exe is in the same folder
as the other dependencies. Optionally, can be run with transcribe2.py as long as
the necessary dependencies from the README have been installed into the same folder.

''' 
backgroundColor = "light yellow" # Also accepts Hex
loadingSummaryText = "Mr.Simpson is generating your summary"
errorText = "Link is not found!"

def vaildYTLink(link):
    if link == "":
        return False
    return True

def buttonClicked(root, textEntry, button):
    if vaildYTLink(textEntry.get()):
        destroyPrevLabel(root)
        textEntry.config(state="disabled")
        button.config(state="disabled")

        loadingLabel = Label(root, text=loadingSummaryText, padx=50, bg=backgroundColor)
        loadingLabel.grid(row=4, column=0)

        link = textEntry.get()

        def process_video():
            # Step 1: Transcribe the video
            # runs transcription script on video and creates transcript.txt
            transcription_result = subprocess.run(
                ["transcribe2.exe", link, "base.en", "transcript.txt"],  #Can optionally run with transcribe2.py
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            print(f"Transcription Output: {transcription_result.stdout}")
            
            if transcription_result.returncode != 0:
                print(f"Error in transcription: {transcription_result.stderr}")
                return

            # Step 2: Summarize the transcript
            # runs summarizer model on the transcript file
            summarize_result = subprocess.run(
                ["python", "summarizer.py", "transcript.txt"],
                capture_output=True, text=True
            )
            print(f"Summarization Output: {summarize_result.stdout}")
            print(f"Summarization Error: {summarize_result.stderr}")

            if summarize_result.returncode != 0:
                print(f"Error in summarization: {summarize_result.stderr}")
                return

            # Step 3: Pass the summary back to the UI
            summaryReady(root, summarize_result.stdout)

            # Re-enable the UI elements after processing
            textEntry.config(state="normal")
            button.config(state="normal")

        # Run the process in a separate thread to prevent UI freeze
        threading.Thread(target=process_video).start()
    else:
        destroyPrevLabel(root)
        errorLabel = Label(root, text=errorText, padx=50, bg=backgroundColor)
        errorLabel.grid(row=4, column=0)

def destroyPrevLabel(root):
    """
    Destroys previous text labels to prevent overlapping and memory leaks
    """
    for label in root.winfo_children():
        if isinstance(label, Label):
            if label.cget('text') == loadingSummaryText or label.cget('text') == errorText:
                label.destroy()

def summaryReady(root, summaryString):
    """
    Shows the YouTube video summary once ready
    """
    destroyPrevLabel(root)
    for label in root.winfo_children():
        if isinstance(label, LabelFrame):
            if label.cget('text') == "Summary":
                label.destroy()
                continue
    frame = LabelFrame(root, text="Summary", padx=20, pady= 20, bg=backgroundColor)
    frame.grid(row=4, column=0)
    summaryLabel = Label(frame, text=summaryString, bg=backgroundColor, anchor="nw", justify="left", wraplength=800)
    summaryLabel.grid(row=0,column=0,sticky="nw")

def main():
    root = Tk()
    root.title("Deep Thinkers' AI summarization model")

    root.minsize(1000, 600)
    root.maxsize(1000,600)

    mainFrame = Frame(root)
    mainFrame.pack(fill=BOTH, expand=1)

    canvas = Canvas(mainFrame, bg="light yellow")
    canvas.pack(side=LEFT, fill=BOTH, expand=1)

    scrollBar = Scrollbar(mainFrame, orient=VERTICAL, command=canvas.yview)
    scrollBar.pack(side=RIGHT, fill=Y)

    canvas.configure(yscrollcommand=scrollBar.set)

    innerFrame = Frame(canvas, width=1000, height=600, bg= backgroundColor)
    innerFrame.pack()

    innerFrame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0,0), window=innerFrame, anchor="nw")

    directionLabel = Label(innerFrame, text="Enter a Youtube link to generate a summary!", padx=36, bg=backgroundColor)
    textEntry = Entry(innerFrame, width=50, borderwidth=5)
    button = Button(innerFrame, text="Generate Summary", command=lambda: buttonClicked(innerFrame, textEntry, button))

    directionLabel.config(font=("Segoe UI", 15))
    textEntry.config(font=("Segoe UI", 12))
    button.config(font=("Segoe UI", 10))

    directionLabel.grid(row=0, column=0, pady=(50, 10), padx=300)
    textEntry.grid(row=1, column=0, pady=10)
    button.grid(row=2, column=0, pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()
