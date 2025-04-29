To download the latest attachment from Outlook emails using a partial subject match, follow this clean and updated code:


---

✅ Python Script to Download Latest Attachment by Partial Subject

import win32com.client
import os

# Config
subject_keyword = "WIP data"  # Partial subject
save_folder = r"C:\Your\Download\Path"  # <-- change to your actual path

# Setup Outlook
outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
inbox = outlook.GetDefaultFolder(6)  # 6 = Inbox
messages = inbox.Items
messages.Sort("[ReceivedTime]", True)  # Sort emails by latest first

# Search and save latest matching attachment
for message in messages:
    if message.Subject and subject_keyword.lower() in message.Subject.lower():
        attachments = message.Attachments
        for attachment in attachments:
            file_path = os.path.join(save_folder, attachment.FileName)
            print("Saving:", file_path)
            attachment.SaveAsFile(file_path)
        break  # Stop after first matching (latest) email


---

Key Features:

Partial match with in and .lower()

Sort by date descending: messages.Sort("[ReceivedTime]", True)

Stops after latest match: break



---

Let me know if your emails are in a subfolder or shared mailbox — I’ll adjust the code accordingly.

