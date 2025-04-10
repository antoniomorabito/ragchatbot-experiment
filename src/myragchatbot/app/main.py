import os
import chainlit as cl

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@cl.on_chat_start
async def start():
    await cl.Message("📤 Please upload your document (PDF/TXT).").send()

    # Ask for file
    files = await cl.AskFileMessage(
        content="Please upload a file to process 👇",
        accept=[".pdf", ".txt"],
        max_size_mb=20,
        max_files=1
    ).send()

    file = files[0]
    file_path = os.path.join(UPLOAD_DIR, file.name)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    await cl.Message(f"✅ File `{file.name}` saved to `{UPLOAD_DIR}/`").send()
