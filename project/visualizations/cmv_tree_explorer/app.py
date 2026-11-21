
"""Interactive CMV Conversation Tree Explorer."""

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlsplit, unquote

import pandas as pd
import plotly.graph_objects as go
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="CMV Conversation Explorer",
    page_icon="🌳",
    layout="wide"
)

COLORS = {
    "OP": "#0c91b5",
    "Challenger": "#e8ae4a",
    "Other": "#9ba8ba",
    "Unknown": "#78818b"
}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_setting(name, default=None):
    """Read Streamlit secrets or .env variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name) or default


def connection_config():
    """
    Support DATABASE_URL or separate DB_* environment variables.
    Supabase session pooler works with port 5432.
    """

    url = get_setting("DATABASE_URL")

    if url and "YOUR_" not in url:
        return {
            "conninfo": url
        }

    host = get_setting("DB_HOST")
    port = get_setting("DB_PORT", "5432")
    dbname = get_setting("DB_NAME", "postgres")
    user = get_setting("DB_USER")
    password = get_setting("DB_PASSWORD")

    if not all([host, user, password]):
        return None

    return {
        "host": host,
        "port": int(port),
        "dbname": dbname,
        "user": user,
        "password": password,
        "sslmode": "require",
        "connect_timeout": 12
    }


def safe_db_error(exc, config):
    """Display errors without revealing database credentials."""

    message = str(exc)
    secrets = []

    if config:
        if config.get("password"):
            secrets.append(str(config["password"]))

        if config.get("conninfo"):
            try:
                parsed = urlsplit(config["conninfo"])
                if parsed.password:
                    secrets.append(parsed.password)
                    secrets.append(unquote(parsed.password))
            except Exception:
                pass

    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")

    message = re.sub(
        r"(postgres(?:ql)?://)[^\s/@]+@",
        r"\1[REDACTED]@",
        message
    )

    return message[:1200]


@st.cache_data(ttl=600, show_spinner="Loading conversation...")
def fetch_post(post_id, connection_items):
    """
    Fetch post, comments, and labeled threads.
    Read-only queries against the cmv schema.
    """

    config = dict(connection_items)

    with psycopg.connect(
        **config
    ) as conn:

        with conn.cursor(row_factory=dict_row) as cur:

            cur.execute("SET TRANSACTION READ ONLY")

            # Original post
            cur.execute(
                """
                SELECT
                    post_id,
                    author,
                    title,
                    created_utc,
                    num_comments
                FROM cmv.posts
                WHERE post_id = %s
                """,
                (post_id,)
            )

            post = cur.fetchone()

            if not post:
                return None, [], []

            # All comments belonging to the post
            cur.execute(
                """
                SELECT
                    comment_id,
                    post_id,
                    parent_comment_id,
                    original_parent_id,
                    author,
                    created_utc,
                    score
                FROM cmv.comments
                WHERE post_id = %s
                ORDER BY created_utc NULLS LAST, comment_id
                """,
                (post_id,)
            )

            comments = cur.fetchall()

            # Positive and negative thread labels
            cur.execute(
                """
                SELECT
                    thread_id,
                    pair_id,
                    outcome,
                    ancestor_id,
                    author
                FROM cmv.threads
                WHERE post_id = %s
                """,
                (post_id,)
            )

            labels = cur.fetchall()

    return post, comments, labels


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def graph_maps(comments):
    """Build parent-child relationships."""

    nodes = {
        comment["comment_id"]: comment
        for comment in comments
    }

    children = defaultdict(list)
    roots = []

    for comment in comments:

        cid = comment["comment_id"]
        parent = comment["parent_comment_id"]

        if parent and parent in nodes:
            children[parent].append(cid)
        else:
            roots.append(cid)

    return nodes, children, roots


def descendants(root, children, nodes, cutoff=None):
    """Return all descendants of a branch root."""

    stack = [(root, 0)]
    output = []
    visited = set()

    while stack:

        cid, depth = stack.pop()

        if cid in visited or cid not in nodes:
            continue

        visited.add(cid)

        comment = nodes[cid]

        # Optional temporal restriction
        if cutoff is not None:

            timestamp = comment["created_utc"]

            if timestamp is None or timestamp > cutoff:
                continue

        output.append((cid, depth))

        for child in reversed(children[cid]):
            stack.append((child, depth + 1))

    return output


# ============================================================
# AUTHOR CLASSIFICATION
# ============================================================

def classify(comment, op, challenger):
    """Classify each comment's author role."""

    author = comment["author"]

    if author is None:
        return "Unknown"

    if author == op:
        return "OP"

    if author == challenger:
        return "Challenger"

    return "Other"


def author_fmt(author):
    return author if author is not None else "[unknown]"


def seconds_fmt(timestamp):
    if timestamp is None:
        return "unknown"

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


# ============================================================
# ENGAGEMENT METRICS
# ============================================================

def alternation(ids, nodes, children, op, challenger):
    """
    Longest consecutive OP/challenger exchange
    along a single parent-child path.
    """

    if not ids:
        return 0

    allowed = set(ids)

    stack = [(ids[0], 0, None)]
    best = 0

    while stack:

        cid, length, previous = stack.pop()

        role = classify(
            nodes[cid],
            op,
            challenger
        )

        if role in ("OP", "Challenger") and role != previous:
            next_length = length + 1

        elif role in ("OP", "Challenger"):
            next_length = 1

        else:
            next_length = 0

        best = max(best, next_length)

        for child in children[cid]:

            if child in allowed:
                stack.append(
                    (child, next_length, role)
                )

    return best


def metrics(ids, nodes, children, op, challenger):
    """Compute structural engagement features."""

    if not ids:
        return {
            "descendants": 0,
            "op_replies": 0,
            "replies_to_op": 0,
            "distinct_authors": 0,
            "alternation": 0
        }

    ids_set = set(ids)

    descendant_count = len(ids) - 1

    op_replies = sum(
        nodes[cid]["author"] == op
        for cid in ids
    )

    replies_to_op = []

    for cid in ids:

        comment = nodes[cid]
        parent = comment["parent_comment_id"]

        if parent not in ids_set:
            continue

        parent_author = nodes[parent]["author"]

        if (
            parent_author == op
            and comment["author"] != op
        ):
            replies_to_op.append(comment)

    distinct_authors = {
        comment["author"]
        for comment in replies_to_op
        if comment["author"] is not None
    }

    return {
        "descendants": descendant_count,
        "op_replies": op_replies,
        "replies_to_op": len(replies_to_op),
        "distinct_authors": len(distinct_authors),
        "alternation": alternation(
            ids, nodes, children, op, challenger
        )
    }


# ============================================================
# TREE LAYOUT
# ============================================================

def layout_tree(root, included, children):
    """Arrange a hierarchical tree from top to bottom."""

    allowed = set(included)
    coords = {}

    leaf_index = 0
    visiting = set()

    def walk(cid, depth):

        nonlocal leaf_index

        if cid in visiting:
            coords[cid] = (leaf_index, -depth)
            leaf_index += 1
            return coords[cid][0]

        visiting.add(cid)

        kids = [
            child
            for child in children[cid]
            if child in allowed and child not in coords
        ]

        if not kids:

            x = leaf_index
            leaf_index += 1

        else:

            xs = [
                walk(child, depth + 1)
                for child in kids
            ]

            x = (min(xs) + max(xs)) / 2

        coords[cid] = (x, -depth)

        visiting.remove(cid)

        return x

    walk(root, 0)

    return coords


# ============================================================
# INTERACTIVE VISUALIZATION
# ============================================================

def make_figure(
    root,
    branch_ids,
    nodes,
    children,
    op,
    challenger,
    label,
    mark_exchanges
):

    coords = layout_tree(
        root,
        branch_ids,
        children
    )

    ids = list(coords.keys())
    allowed = set(ids)

    regular_x = []
    regular_y = []

    exchange_x = []
    exchange_y = []

    # Draw parent-child connections
    for cid in ids:

        parent = nodes[cid]["parent_comment_id"]

        if parent not in allowed:
            continue

        x1, y1 = coords[parent]
        x2, y2 = coords[cid]

        parent_role = classify(
            nodes[parent],
            op,
            challenger
        )

        child_role = classify(
            nodes[cid],
            op,
            challenger
        )

        is_exchange = (
            mark_exchanges
            and {parent_role, child_role}
            == {"OP", "Challenger"}
        )

        if is_exchange:
            xx, yy = exchange_x, exchange_y
        else:
            xx, yy = regular_x, regular_y

        xx.extend([x1, x2, None])
        yy.extend([y1, y2, None])

    fig = go.Figure()

    # Regular connections
    fig.add_trace(
        go.Scatter(
            x=regular_x,
            y=regular_y,
            mode="lines",
            line=dict(
                color="#8894a6",
                width=1.4
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Direct OP-challenger exchanges
    fig.add_trace(
        go.Scatter(
            x=exchange_x,
            y=exchange_y,
            mode="lines",
            line=dict(
                color="#d69c45",
                width=3.3
            ),
            hoverinfo="skip",
            name="Direct OP ↔ challenger"
        )
    )

    # Comment nodes
    for role in (
        "OP",
        "Challenger",
        "Other",
        "Unknown"
    ):

        role_ids = [
            cid
            for cid in ids
            if classify(
                nodes[cid],
                op,
                challenger
            ) == role
        ]

        if not role_ids:
            continue

        texts = [
            (
                f"{cid}<br>"
                f"u/{author_fmt(nodes[cid]['author'])}<br>"
                f"{seconds_fmt(nodes[cid]['created_utc'])}"
            )
            for cid in role_ids
        ]

        fig.add_trace(
            go.Scatter(
                x=[coords[cid][0] for cid in role_ids],
                y=[coords[cid][1] for cid in role_ids],
                mode="markers",
                name=role,
                marker=dict(
                    color=COLORS[role],
                    size=13 if role != "Other" else 9,
                    line=dict(
                        color="#ffffff",
                        width=1.2
                    )
                ),
                customdata=role_ids,
                text=texts,
                hovertemplate="%{text}<extra></extra>"
            )
        )

    # Winning branch marker
    # This identifies the winning branch root,
    # NOT the exact delta-awarding comment.
    if label == "positive":

        x, y = coords[root]

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y + 0.34],
                mode="markers",
                name="Winning branch root",
                marker=dict(
                    symbol="star",
                    color="#de3461",
                    size=18
                ),
                hovertemplate=(
                    "Winning branch label<br>"
                    "Exact delta event not available"
                    "<extra></extra>"
                )
            )
        )

    max_depth = max(
        -coords[cid][1]
        for cid in ids
    )

    fig.update_layout(
        height=max(
            550,
            min(1200, 100 + 95 * (1 + max_depth))
        ),
        margin=dict(
            l=15,
            r=15,
            t=20,
            b=15
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=-0.03,
            x=0
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_color="white"
        ),
        clickmode="event+select"
    )

    return fig


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("🌳 CMV Conversation Tree Explorer")

st.caption(
    "Explore conversation branches, OP engagement, "
    "and parent-child relationships."
)

# Database configuration
cfg = connection_config()

if not cfg:
    st.error(
        "Database configuration missing. "
        "Set DATABASE_URL or DB_HOST, DB_USER, "
        "and DB_PASSWORD in your .env file."
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar: Post lookup
# ------------------------------------------------------------

with st.sidebar:

    st.header("Find a conversation")

    post_id = st.text_input(
        "Reddit post ID",
        value="t3_18dj9h",
        help="Example: t3_18dj9h"
    ).strip()

    if st.button("Refresh cached post"):
        fetch_post.clear()


# ------------------------------------------------------------
# Fetch data
# ------------------------------------------------------------

try:

    post, comments, labels = fetch_post(
        post_id,
        tuple(sorted(cfg.items()))
    )

except psycopg.OperationalError as exc:

    st.error("Database connection failed.")

    st.code(
        safe_db_error(exc, cfg),
        language="text"
    )

    st.info(
        "Check your Supabase username, password, "
        "pooler hostname, port, and network."
    )

    st.stop()

except psycopg.Error as exc:

    st.error("Database query failed.")

    st.code(
        safe_db_error(exc, cfg),
        language="text"
    )

    st.info(
        "Check SELECT permissions on the cmv schema."
    )

    st.stop()

except Exception as exc:

    st.error(
        f"Unexpected error: {type(exc).__name__}"
    )

    st.code(
        safe_db_error(exc, cfg),
        language="text"
    )

    st.stop()


if not post:

    st.warning(
        f"Post {post_id} was not found."
    )

    st.stop()


# ------------------------------------------------------------
# Construct graph
# ------------------------------------------------------------

nodes, children, roots = graph_maps(comments)

labels_by_root = defaultdict(set)

for thread in labels:

    labels_by_root[
        thread["ancestor_id"]
    ].add(thread["outcome"])


st.subheader(
    post.get("title") or post_id
)

st.caption(
    f"Post: {post_id} | "
    f"OP: u/{author_fmt(post['author'])} | "
    f"Comments: {len(comments):,} | "
    f"Branches: {len(roots):,}"
)

if not comments:

    st.info("No comments found.")

    st.stop()


# ------------------------------------------------------------
# Root selection
# ------------------------------------------------------------

root_options = sorted(
    roots,
    key=lambda cid: (
        nodes[cid]["created_utc"] is None,
        nodes[cid]["created_utc"] or 0
    )
)


def root_display(root):

    comment = nodes[root]

    outcomes = labels_by_root.get(
        root,
        set()
    )

    if "positive" in outcomes:
        flag = "⭐ Winning"

    elif "negative" in outcomes:
        flag = "Negative"

    else:
        flag = "Unlabeled"

    return (
        f"{root} | "
        f"u/{author_fmt(comment['author'])} | "
        f"{flag}"
    )


with st.sidebar:

    st.divider()

    st.header("Select branch")

    only_labeled = st.checkbox(
        "Only labeled branches",
        value=False
    )

    if only_labeled:

        options = [
            root
            for root in root_options
            if root in labels_by_root
        ]

    else:
        options = root_options

    if not options:

        st.warning("No matching branches.")
        st.stop()

    preferred = next(
        (
            i
            for i, root in enumerate(options)
            if "positive" in labels_by_root.get(
                root, set()
            )
        ),
        0
    )

    root = st.selectbox(
        "Top-level reply",
        options,
        index=preferred,
        format_func=root_display
    )

    st.divider()

    # Observation window
    st.header("Observation window")

    window = st.selectbox(
        "Comments to include",
        [
            "Full archive",
            "First 15 minutes",
            "First hour",
            "First 6 hours",
            "First 24 hours",
            "First 7 days"
        ]
    )

    window_seconds = {
        "Full archive": None,
        "First 15 minutes": 900,
        "First hour": 3600,
        "First 6 hours": 21600,
        "First 24 hours": 86400,
        "First 7 days": 604800
    }

    duration = window_seconds[window]

    cutoff = (
        None
        if duration is None
        else post["created_utc"] + duration
    )

    mark_exchanges = st.checkbox(
        "Highlight OP–challenger exchanges",
        value=True
    )

    max_nodes = st.slider(
        "Maximum nodes to draw",
        min_value=50,
        max_value=1500,
        value=400,
        step=50
    )


# ------------------------------------------------------------
# Selected branch
# ------------------------------------------------------------

visible_branch = descendants(
    root,
    children,
    nodes,
    cutoff
)

if not visible_branch:

    st.warning(
        "This branch had not opened during "
        "the selected observation window."
    )

    st.stop()


branch_ids = [
    cid
    for cid, depth in visible_branch
]

render_ids = branch_ids[:max_nodes]

challenger = nodes[root]["author"]

label_set = labels_by_root.get(
    root,
    set()
)

if "positive" in label_set:
    label = "positive"

elif "negative" in label_set:
    label = "negative"

else:
    label = "unlabeled"


# ------------------------------------------------------------
# Branch outcome
# ------------------------------------------------------------

if label == "positive":

    st.success(
        "⭐ Positive/winning branch. "
        "The star identifies the labeled branch root, "
        "not the exact delta-awarding comment."
    )

elif label == "negative":

    st.info(
        "This branch has a negative label "
        "in the paired CMV dataset."
    )

else:

    st.caption(
        "This branch is not labeled "
        "in the paired dataset."
    )


# ------------------------------------------------------------
# Engagement statistics
# ------------------------------------------------------------

m = metrics(
    branch_ids,
    nodes,
    children,
    post["author"],
    challenger
)

cols = st.columns(5)

metric_names = [
    "Descendants",
    "OP replies",
    "Replies to OP",
    "Distinct authors",
    "OP–challenger alternation"
]

metric_values = [
    m["descendants"],
    m["op_replies"],
    m["replies_to_op"],
    m["distinct_authors"],
    m["alternation"]
]

for col, name, value in zip(
    cols,
    metric_names,
    metric_values
):

    col.metric(name, value)


st.caption(
    "Alternation measures the longest consecutive "
    "OP–challenger exchange on a parent-child path."
)


# ------------------------------------------------------------
# Tree and comment inspector
# ------------------------------------------------------------

left, right = st.columns(
    [3.4, 1.35],
    gap="large"
)

with left:

    st.markdown("### Parent → Child Conversation Tree")

    if len(branch_ids) > max_nodes:

        st.warning(
            f"Displaying {max_nodes} of "
            f"{len(branch_ids)} comments."
        )

    fig = make_figure(
        root,
        render_ids,
        nodes,
        children,
        post["author"],
        challenger,
        label,
        mark_exchanges
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=f"tree_{post_id}_{root}_{window}"
    )

    selected = None

    try:

        for point in event.selection.points:

            cid = point.get("customdata")

            if isinstance(cid, str) and cid in nodes:

                selected = cid
                break

    except (AttributeError, TypeError):
        pass


with right:

    st.markdown("### Inspect Comment")

    all_ids = set(branch_ids)

    selected_index = (
        branch_ids.index(selected)
        if selected in all_ids
        else 0
    )

    cid = st.selectbox(
        "Select node (or click a dot)",
        branch_ids,
        index=selected_index,
        format_func=lambda value: (
            f"{value} | "
            f"{author_fmt(nodes[value]['author'])}"
        )
    )

    comment = nodes[cid]

    role = classify(
        comment,
        post["author"],
        challenger
    )

    st.write(f"**Role:** {role}")

    st.write(
        f"**Author:** "
        f"u/{author_fmt(comment['author'])}"
    )

    st.write(
        f"**Created:** "
        f"{seconds_fmt(comment['created_utc'])}"
    )

    st.write(
        f"**Score:** "
        f"{comment['score']}"
    )

    st.code(cid)

    parent = comment["parent_comment_id"]

    st.write(
        f"**Parent:** `{parent or post_id}`"
    )

    child_count = sum(
        child in all_ids
        for child in children[cid]
    )

    st.write(
        f"**Direct children:** {child_count}"
    )

    # Ancestor path
    path = []
    visited = set()
    current = cid

    while (
        current in nodes
        and current not in visited
    ):

        visited.add(current)
        path.append(current)

        current = nodes[current][
            "parent_comment_id"
        ]

    st.markdown("**Ancestor path:**")

    st.caption(
        " → ".join(reversed(path))
    )

    st.caption(
        "Comment text is not stored "
        "in the current metadata table."
    )

    if cid.startswith("t1_") and post_id.startswith("t3_"):

        url = (
            "https://www.reddit.com/r/changemyview/"
            f"comments/{post_id[3:]}/_/{cid[3:]}/"
        )

        st.link_button(
            "Open on Reddit",
            url
        )


# ============================================================
# ALL BRANCHES COMPARISON
# ============================================================

st.divider()

st.markdown(
    "### Compare All Top-Level Branches"
)

summary = []

for branch_root in root_options:

    branch = descendants(
        branch_root,
        children,
        nodes,
        cutoff
    )

    ids = [
        cid
        for cid, depth in branch
    ]

    if not ids:
        continue

    branch_challenger = nodes[
        branch_root
    ]["author"]

    branch_metrics = metrics(
        ids,
        nodes,
        children,
        post["author"],
        branch_challenger
    )

    outcomes = labels_by_root.get(
        branch_root,
        set()
    )

    summary.append({
        "root_id": branch_root,
        "challenger": author_fmt(
            branch_challenger
        ),
        "label": (
            ", ".join(sorted(outcomes))
            if outcomes
            else "unlabeled"
        ),
        "opening_utc": seconds_fmt(
            nodes[branch_root]["created_utc"]
        ),
        **branch_metrics
    })


df = pd.DataFrame(summary)

st.dataframe(
    df,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander("Methodological limitations"):

    st.write(
        """
        This visualization is an exploratory research tool.

        Positive labels identify winning branches, not
        the precise comment where a delta was awarded.

        Full-archive engagement measurements may include
        activity after the delta event.

        Observation windows are measured relative to the
        original post's creation time.

        Unlabeled branches should not automatically be
        interpreted as non-winning examples.

        The graph reveals conversational structure and
        participation patterns, not causal effects.
        """
    )
