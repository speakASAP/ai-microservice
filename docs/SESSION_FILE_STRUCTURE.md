# Corrected File Structure for StateX Multi-Agent System

## Session Directory Structure

```text
data/uploads/{user_id}/sess_{timestamp}_{random}/
├── form_data.md                  # ← NLP Agent reads this
├── nlp.md                        # ← Summarizer Agent reads this
├── voicerecording.md             # ← Summarizer Agent reads this  
├── attachments.md                # ← Summarizer Agent reads this
├── prototype.md                  # ← Prototype agent saves this
├── summary.md                    # ← Final summary saved here
└── files/                        # ← Subdirectory for actual files
    ├── {file_id}.webm            # ← ASR Agent processes this
    ├── {file_id}.pdf             # ← Document Agent processes this
    ├── {file_id}.docx            # ← Document Agent processes this
    └── {file_id}.txt             # ← Document Agent processes this
```

## Agent File Reading Logic

### 1. **NLP Agent**

- **Reads from**: `form_data.md` (session root)
- **Purpose**: Analyzes the form description and requirements
- **Input**: Text content from form submission

### 2. **ASR Agent**

- **Reads from**: `files/` subdirectory
- **File types**: `.webm`, `.mp3`, `.wav`, `.m4a`, `.ogg`
- **Purpose**: Transcribes voice recordings to text
- **Input**: Audio files from voice recording

### 3. **Document Agent**

- **Reads from**: `files/` subdirectory  
- **File types**: `.pdf`, `.doc`, `.docx`, `.txt`, `.rtf`, `.odt`
- **Purpose**: Extracts and analyzes document content
- **Input**: Attachment files from form submission

### 4. **Summarizer Agent**

- **Reads from**: Session root directory
- **Files**: `nlp.md`, `voicerecording.md`, `attachments.md`
- **Purpose**: Aggregates all agent results into comprehensive summary
- **Input**: Results from all other agents

### 5. **Prototype Agent**

- **Reads from**: Summarizer results
- **Purpose**: Generates prototype based on analysis
- **Input**: Summary and analysis data

## File Path Examples

Based on the actual session: `sess_1760089926_c3abd1aa`

```text
data/uploads/6965c04f3439e47fce82274d644faaa3/sess_1760089926_c3abd1aa/
├── form_data.md
├── nlp.md
├── voicerecording.md  
├── attachments.md
├── prototype.md
├── summary.md
└── files/
    ├── 3276b2f589924a37bd69fe32798914e8.webm  # Voice recording
    └── f2b6ed9762f8446f87e723ed860fac2a.pdf   # PDF attachment
```

## Key Changes Made

1. **Fixed file structure paths** - Voice files and attachments are now correctly read from `files/` subdirectory
2. **Enhanced agent input data** - All agents now receive `submission_id`, `user_id`, and `read_from_disk` flag
3. **Improved disk reading functions** - Added support for subdirectory reading
4. **Comprehensive Telegram notifications** - Added detailed input/output data, execution time, confidence scores, and error messages
5. **Agent-specific formatting** - Each agent type has specialized notification formatting

## Agent Workflow

1. **Form Submission** → Files saved to `files/` subdirectory
2. **NLP Agent** → Reads `form_data.md` → Saves results to `nlp.md`
3. **ASR Agent** → Reads audio files from `files/` → Saves results to `voicerecording.md`
4. **Document Agent** → Reads document files from `files/` → Saves results to `attachments.md`
5. **Summarizer Agent** → Reads all result files → Saves summary to `summary.md`
6. **Prototype Agent** → Uses summary data → Saves results to `prototype.md`
7. **Telegram Notifications** → Sent for each agent with comprehensive data
