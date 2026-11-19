# CMV Conversation Tree Explorer

A read-only interactive explorer for the existing `cmv` schema in Supabase.

## Run

```bash
cd cmv_tree_explorer
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your database connection string.
streamlit run app.py
```

Use the Supabase Dashboard **Connect** button to get a Postgres connection string. A read-only database role with SELECT access to `cmv.posts`, `cmv.comments`, and `cmv.threads` is preferable. The app never writes to the database. Do not commit `.env` to Git. If connecting via a pooler, use the connection string provided by Supabase and ensure SSL is enabled.

Enter a post ID (`t3_18dj9h` is prefilled); select any top-level branch (a reply whose `parent_comment_id` is null). The app retrieves the complete parent-child structure for the post, then visualizes just that branch.

## Interactions

- Choose post ID, top-level branch, timeline cutoff, and graph display mode.
- Click a node in the interactive Plotly tree to inspect metadata, parent, children, and the comment's path up to the root.
- Toggle highlighting of OP-to-challenger edges and hide comments beyond the time cutoff.
- View metrics: descendants, OP replies, replies to OP, distinct authors replying to OP, and longest same-challenger alternation on a root-to-leaf path.
- View matching positive/negative dataset thread labels, clearly distinguished from *observed delta comments*.
- Compare the available top-level branches via the sortable summary table.

## Correct interpretation

The `cmv.threads.outcome = 'positive'` label is associated with a **branch ancestor**. The loaded metadata does **not** contain a per-comment delta award event or timestamp. The app therefore marks winning branch roots with a star, not a precise delta-award node. Never infer delta timing from this alone.

Time filters measure elapsed time since the post was created. Missing timestamps are excluded when filtering. A non-winning branch is not proof that its arguments were ineffective. Full-archive graph metrics are **retrospective** and may contain post-outcome activity; do not use them unmodified for delta prediction.

## SQL and performance

The app runs parameterized, read-only SELECT queries. It loads all comments for one post once, cached for 10 minutes. Very large posts may be slower; default layout caps rendered nodes and shows a warning. No post content/body is fetched because the current `cmv.comments` table contains metadata but not text.
