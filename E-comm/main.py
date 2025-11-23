from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv
import logfire

from fasthtml.common import *  # FastHTML components, fast_app, serve, etc.
from pydantic_ai import Agent, RunContext, ModelMessage

# ---------------------------------------------------------------------------
# Environment & Logfire configuration
# ---------------------------------------------------------------------------

# Load .env so GOOGLE_API_KEY / GEMINI_API_KEY etc. are available
load_dotenv()

# Configure Logfire. It will read LOGFIRE_TOKEN / .logfire if present.
logfire.configure()

# Instrument Pydantic-AI so every agent run, tool call, and token usage is traced.
logfire.instrument_pydantic_ai()

# ---------------------------------------------------------------------------
# Simple product catalog for shoes
# ---------------------------------------------------------------------------


@dataclass
class Product:
    id: str
    brand: str
    name: str
    price: int  # price in rupees for demo
    image_url: str  # product image URL


PRODUCTS: Dict[str, Product] = {
    "nike_pegasus": Product(
        id="nike_pegasus",
        brand="Nike",
        name="Nike Air Zoom Pegasus",
        price=9999,
        image_url="https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/99486859-0ff3-46b4-949b-2d16af2ad421/air-zoom-pegasus-40-running-shoes-PX6r8S.png",
    ),
    "adidas_ultraboost": Product(
        id="adidas_ultraboost",
        brand="Adidas",
        name="Adidas Ultraboost Light",
        price=11999,
        image_url="https://via.placeholder.com/300x200/000000/FFFFFF?text=Adidas+Ultraboost",
    ),
    "puma_velocity": Product(
        id="puma_velocity",
        brand="Puma",
        name="Puma Velocity Nitro",
        price=8999,
        image_url="https://via.placeholder.com/300x200/FF6B35/FFFFFF?text=Puma+Velocity",
    ),
    "asics_nimbus": Product(
        id="asics_nimbus",
        brand="ASICS",
        name="ASICS Gel-Nimbus",
        price=10999,
        image_url="https://via.placeholder.com/300x200/0066CC/FFFFFF?text=ASICS+Nimbus",
    ),
    "skechers_go_run": Product(
        id="skechers_go_run",
        brand="Skechers",
        name="Skechers GO Run Ride",
        price=7499,
        image_url="https://via.placeholder.com/300x200/FF1493/FFFFFF?text=Skechers+GO+Run",
    ),
    "nb_1080": Product(
        id="nb_1080",
        brand="New Balance",
        name="New Balance Fresh Foam 1080",
        price=10499,
        image_url="https://nb.scene7.com/is/image/NB/m1080b12_nb_02_i?$dw_detail_main_lg$&bgc=f1f1f1&layer=1&bgcolor=f1f1f1&blendMode=mult&scale=10&wid=1600&hei=1600",
    ),
}


def find_product_by_brand(brand: str) -> Optional[Product]:
    """Utility: pick a default product for a given brand (1 SKU per brand)."""
    brand_lower = brand.strip().lower()
    for p in PRODUCTS.values():
        if p.brand.lower() == brand_lower:
            return p
    return None


# ---------------------------------------------------------------------------
# Cart state & store (stateful across requests)
# ---------------------------------------------------------------------------


@dataclass
class CartItem:
    product_id: str
    brand: str
    name: str
    quantity: int
    unit_price: int

    @property
    def subtotal(self) -> int:
        return self.quantity * self.unit_price


@dataclass
class Cart:
    # key = product_id
    items: Dict[str, CartItem] = field(default_factory=dict)

    def total_quantity(self) -> int:
        return sum(i.quantity for i in self.items.values())

    def total_price(self) -> int:
        return sum(i.subtotal for i in self.items.values())


@dataclass
class CartStore:
    """Simple in-memory store, keyed by session_id."""

    carts: Dict[str, Cart] = field(default_factory=dict)

    def get_cart(self, session_id: str) -> Cart:
        if session_id not in self.carts:
            self.carts[session_id] = Cart()
        return self.carts[session_id]


# Single demo user/session for this example.
SESSION_ID = "demo-session"
cart_store = CartStore()

# Chat history for the agent (for model context), separate from UI transcript.
agent_message_history: List[ModelMessage] | None = None

# UI transcript for rendering chat bubbles
chat_transcript: List[Dict[str, str]] = []


# ---------------------------------------------------------------------------
# Pydantic-AI Agent for the cart, with dependencies
# ---------------------------------------------------------------------------


@dataclass
class CartDeps:
    """Dependencies passed to the agent each run."""

    cart_store: CartStore
    session_id: str


cart_agent = Agent(
    "google-gla:gemini-2.5-flash",
    deps_type=CartDeps,
    system_prompt=(
        "You are a state-aware e-commerce assistant for a sneaker store.\n"
        "- Available brands: Nike, Adidas, Puma, ASICS, Skechers, New Balance.\n"
        "- The user can add shoes to their cart, change quantities, remove items,\n"
        "  and ask what is currently in the cart.\n"
        "- Use the tools add_to_cart, update_quantity, remove_from_cart, and view_cart\n"
        "  to keep the underlying cart state in sync with your answers.\n"
        "- ALWAYS call view_cart before answering questions about what is in the cart.\n"
        "- Be explicit when you change state, e.g. 'I added 2 Nike shoes to your cart'.\n"
        "- Keep answers short, friendly, and focused on the cart.\n"
    ),
)


# ---------------------------------------------------------------------------
# Tools: these directly manipulate CartStore state via deps
# ---------------------------------------------------------------------------


@cart_agent.tool
def add_to_cart(
    ctx: RunContext[CartDeps],
    brand: str,
    quantity: int = 1,
) -> str:
    """
    Add shoes of the given brand to the user's cart.

    If the brand already exists in the cart, increase the quantity.
    This uses a default SKU per brand from the product catalog.
    """
    quantity = max(1, int(quantity))

    product = find_product_by_brand(brand)
    if product is None:
        return (
            f"I couldn't find a shoe for brand {brand!r}. "
            "Valid brands are Nike, Adidas, Puma, ASICS, Skechers, and New Balance."
        )

    cart = ctx.deps.cart_store.get_cart(ctx.deps.session_id)
    if product.id in cart.items:
        cart.items[product.id].quantity += quantity
    else:
        cart.items[product.id] = CartItem(
            product_id=product.id,
            brand=product.brand,
            name=product.name,
            quantity=quantity,
            unit_price=product.price,
        )

    return (
        f"Added {quantity}x {product.brand} ({product.name}) to the cart. "
        f"New quantity for this item: {cart.items[product.id].quantity}."
    )


@cart_agent.tool
def update_quantity(
    ctx: RunContext[CartDeps],
    brand: str,
    quantity: int,
) -> str:
    """
    Set the quantity for a given brand.

    If quantity <= 0, the item is removed from the cart.
    """
    product = find_product_by_brand(brand)
    if product is None:
        return (
            f"I couldn't find a shoe for brand {brand!r}. "
            "Valid brands are Nike, Adidas, Puma, ASICS, Skechers, and New Balance."
        )

    cart = ctx.deps.cart_store.get_cart(ctx.deps.session_id)

    if product.id not in cart.items:
        return f"There is no {product.brand} item in the cart to update."

    if quantity <= 0:
        del cart.items[product.id]
        return f"Removed {product.brand} ({product.name}) from the cart."
    else:
        cart.items[product.id].quantity = quantity
        return (
            f"Updated {product.brand} ({product.name}) quantity "
            f"to {quantity} in the cart."
        )


@cart_agent.tool
def remove_from_cart(
    ctx: RunContext[CartDeps],
    brand: str,
) -> str:
    """
    Remove all shoes of the given brand from the cart.
    """
    product = find_product_by_brand(brand)
    if product is None:
        return (
            f"I couldn't find a shoe for brand {brand!r}. "
            "Valid brands are Nike, Adidas, Puma, ASICS, Skechers, and New Balance."
        )

    cart = ctx.deps.cart_store.get_cart(ctx.deps.session_id)
    if product.id not in cart.items:
        return f"There is no {product.brand} item in the cart to remove."

    del cart.items[product.id]
    return f"Removed {product.brand} ({product.name}) from the cart."


@cart_agent.tool
def view_cart(ctx: RunContext[CartDeps]) -> str:
    """
    Return a human-readable summary of the current cart.

    This is the 'state introspection' tool: the agent calls this to see
    what's in the cart before answering questions.
    """
    cart = ctx.deps.cart_store.get_cart(ctx.deps.session_id)
    if not cart.items:
        return "The cart is currently empty."

    lines: List[str] = []
    lines.append("Current cart contents:")
    for item in cart.items.values():
        lines.append(
            f"- {item.brand} ({item.name}): {item.quantity} pair(s), "
            f"₹{item.unit_price} each, subtotal ₹{item.subtotal}"
        )
    lines.append(f"Total items: {cart.total_quantity()}")
    lines.append(f"Cart total: ₹{cart.total_price()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastHTML app & layout helpers
# ---------------------------------------------------------------------------

# Minimal app with PicoCSS (via fast_app) – nice default styling.
app, rt = fast_app()


def render_cart(cart: Cart) -> FastTag:
    """Render the current cart as a nice card with manual controls."""
    if not cart.items:
        body = P(
            "Your cart is empty. Add some shoes by clicking 'Add to Cart' below or ask the assistant ✨",
            cls="muted",
        )
    else:
        cart_items = []
        for item in cart.items.values():
            product = PRODUCTS[item.product_id]
            cart_items.append(
                Article(
                    Div(
                        Img(
                            src=product.image_url,
                            alt=f"{product.brand} {product.name}",
                            cls="cart-item-image",
                        ),
                        Div(
                            H4(item.brand),
                            P(item.name, cls="muted"),
                            P(f"₹{item.unit_price} each", cls="price"),
                            cls="cart-item-info",
                        ),
                        cls="cart-item-header",
                    ),
                    Div(
                        Div(
                            Button(
                                "−",
                                hx_post=f"/cart/decrease/{item.product_id}",
                                hx_target="#cart-container",
                                hx_swap="outerHTML",
                                cls="qty-btn qty-minus",
                            ),
                            Span(f"{item.quantity}", cls="qty-display"),
                            Button(
                                "+",
                                hx_post=f"/cart/increase/{item.product_id}",
                                hx_target="#cart-container",
                                hx_swap="outerHTML",
                                cls="qty-btn qty-plus",
                            ),
                            cls="qty-controls",
                        ),
                        Button(
                            "Remove",
                            hx_post=f"/cart/remove/{item.product_id}",
                            hx_target="#cart-container",
                            hx_swap="outerHTML",
                            cls="remove-btn",
                        ),
                        cls="cart-item-actions",
                    ),
                    P(f"Subtotal: ₹{item.subtotal}", cls="cart-item-subtotal"),
                    cls="cart-item-card",
                )
            )

        body = Div(
            *cart_items,
            Hr(),
            P(
                f"Total ({cart.total_quantity()} item(s)): ₹{cart.total_price()}",
                cls="cart-total",
            ),
        )

    return Article(H2("Your Cart"), body, cls="cart-card", id="cart-container")


def render_products() -> FastTag:
    """Render static product list with images and Add to Cart buttons."""
    cards = []
    for p in PRODUCTS.values():
        cards.append(
            Article(
                Img(src=p.image_url, alt=f"{p.brand} {p.name}", cls="product-image"),
                H3(p.brand),
                P(p.name, cls="muted"),
                P(f"₹{p.price}", cls="product-price"),
                Button(
                    "Add to Cart",
                    hx_post=f"/cart/add/{p.id}",
                    hx_target="#cart-container",
                    hx_swap="outerHTML",
                    cls="btn-primary add-to-cart-btn",
                ),
                P("Or ask the assistant to add this by brand name.", cls="hint"),
                cls="product-card",
            )
        )
    return Section(
        H2("Sneakers Catalog"),
        Div(*cards, cls="products-grid"),
        cls="products-section",
    )


def render_chat(messages: List[Dict[str, str]]) -> FastTag:
    """Render simple chat bubbles for user + assistant messages."""
    if not messages:
        body = P(
            "Start by telling me what kind of shoes you want, e.g.\n"
            '"Add 2 pairs of Nike running shoes to my cart."',
            cls="muted",
        )
    else:
        rendered = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            bubble_cls = (
                "chat-bubble chat-user"
                if role == "user"
                else "chat-bubble chat-assistant"
            )
            label = "You" if role == "user" else "Assistant"
            rendered.append(
                Div(
                    Small(label, cls="chat-label"),
                    P(content),
                    cls=bubble_cls,
                )
            )
        body = Div(*rendered, cls="chat-scroll")

    return Article(
        H2("Cart Assistant"),
        body,
        Form(action=shop, method="post")(
            Label("Message", _for="question"),
            Textarea(
                "",
                id="question",
                name="question",
                placeholder=(
                    "Examples:\n"
                    "- Add 1 pair of Adidas shoes\n"
                    "- Increase Nike to 3 pairs\n"
                    "- Remove Puma from my cart\n"
                    "- What's in my cart now?"
                ),
                required=True,
            ),
            Div(
                Span(
                    "The assistant will call tools to keep your cart in sync.",
                    cls="muted",
                ),
                Button("Send →", type="submit", cls="btn-primary"),
                cls="chat-footer",
            ),
        ),
        cls="chat-card",
    )


def shop_page(
    usage: Optional[dict] = None,
    error: Optional[str] = None,
) -> tuple:
    """Compose the full /shop page."""
    cart = cart_store.get_cart(SESSION_ID)

    # Optional token usage pill
    if usage and usage.get("input_tokens") is not None:
        usage_ft = Div(
            Span("Tokens", cls="muted"),
            Span(f"in: {usage['input_tokens']}", cls="mono"),
            Span("·", cls="muted"),
            Span(f"out: {usage['output_tokens']}", cls="mono"),
            cls="token-pill",
        )
    else:
        usage_ft = Span("Run a message to see token usage.", cls="muted")

    # Enhanced CSS for product images and cart controls
    css = Style(
        """
        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
            gap: 1.5rem;
            align-items: start;
        }
        @media (max-width: 900px) {
            .layout {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
        }
        .product-card, .cart-card, .chat-card {
            border-radius: 1rem;
            padding: 1rem 1.25rem;
        }
        .product-image, .cart-item-image {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .cart-item-image {
            height: 80px;
            width: 80px;
            flex-shrink: 0;
        }
        .product-price {
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--pico-primary);
        }
        .add-to-cart-btn {
            width: 100%;
            margin: 0.5rem 0;
        }
        .cart-item-card {
            border: 1px solid var(--pico-muted-border-color);
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .cart-item-header {
            display: flex;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }
        .cart-item-info h4 {
            margin: 0 0 0.25rem 0;
            font-size: 1.1rem;
        }
        .cart-item-info p {
            margin: 0.25rem 0;
        }
        .cart-item-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin: 0.75rem 0;
        }
        .qty-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .qty-btn {
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1rem;
        }
        .qty-minus {
            background: var(--pico-secondary);
        }
        .qty-plus {
            background: var(--pico-primary);
        }
        .qty-display {
            font-weight: 600;
            min-width: 2rem;
            text-align: center;
            font-size: 1.1rem;
        }
        .remove-btn {
            background: var(--pico-color-red-600);
            color: white;
            border: none;
            padding: 0.25rem 0.75rem;
            border-radius: 0.25rem;
            font-size: 0.9rem;
        }
        .cart-item-subtotal {
            font-weight: 600;
            margin: 0.5rem 0 0 0;
            text-align: right;
        }
        .cart-item + .cart-item {
            margin-top: 0.5rem;
        }
        .cart-total {
            font-weight: 600;
            font-size: 1.1rem;
            text-align: center;
            padding: 0.75rem;
            background: var(--pico-primary-soft);
            border-radius: 0.5rem;
        }
        .chat-scroll {
            max-height: 320px;
            overflow-y: auto;
            padding-right: 0.25rem;
            margin-bottom: 0.75rem;
        }
        .chat-bubble {
            border-radius: 0.75rem;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
        }
        .chat-user {
            background: var(--pico-primary-soft);
            margin-left: 2rem;
        }
        .chat-assistant {
            background: var(--pico-muted-border-color);
            margin-right: 2rem;
        }
        .chat-label {
            font-size: 0.75rem;
            opacity: 0.7;
        }
        .chat-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-top: 0.5rem;
        }
        .hint {
            font-size: 0.85rem;
            opacity: 0.7;
        }
        .header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.8rem;
            background: var(--pico-muted-border-color);
        }
        .pill-dot {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            background: var(--pico-primary);
        }
        .token-pill {
            display: inline-flex;
            gap: 0.4rem;
            font-size: 0.8rem;
            align-items: center;
        }
        .mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
        """
    )

    error_block = (
        Article(
            P(error, cls="muted"),
            cls="cart-card",
        )
        if error
        else None
    )

    return (
        Title("State-Aware Sneaker Cart · Pydantic-AI + FastHTML"),
        Script(src="https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js"),
        css,
        Main(
            Header(
                Div(
                    H1("Sneaker Cart Assistant"),
                    P(
                        "State-aware cart managed by a Pydantic-AI agent using tools.",
                        cls="muted",
                    ),
                    cls="header-row",
                ),
                Div(
                    Span(cls="pill-dot"),
                    Span("Tool calls & tokens visible in Logfire"),
                    cls="pill",
                ),
                cls="header-row",
            ),
            Section(
                Div(
                    Div(
                        render_products(),
                        render_cart(cart),
                        cls="left-column",
                    ),
                    Div(
                        render_chat(chat_transcript),
                        Article(
                            H3("Run usage"),
                            usage_ft,
                            cls="cart-card",
                        ),
                        cls="right-column",
                    ),
                    cls="layout",
                ),
                error_block if error_block else "",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Route: /shop  (GET shows UI, POST runs the agent & updates state)
# ---------------------------------------------------------------------------


# Cart manipulation routes for manual controls
@rt("/cart/add/{product_id}")
def add_to_cart_route(product_id: str):
    """Add a product to cart manually."""
    if product_id not in PRODUCTS:
        return render_cart(cart_store.get_cart(SESSION_ID))

    product = PRODUCTS[product_id]
    cart = cart_store.get_cart(SESSION_ID)

    if product_id in cart.items:
        cart.items[product_id].quantity += 1
    else:
        cart.items[product_id] = CartItem(
            product_id=product_id,
            brand=product.brand,
            name=product.name,
            quantity=1,
            unit_price=product.price,
        )

    return render_cart(cart)


@rt("/cart/increase/{product_id}")
def increase_quantity_route(product_id: str):
    """Increase quantity of a product in cart."""
    cart = cart_store.get_cart(SESSION_ID)
    if product_id in cart.items:
        cart.items[product_id].quantity += 1
    return render_cart(cart)


@rt("/cart/decrease/{product_id}")
def decrease_quantity_route(product_id: str):
    """Decrease quantity of a product in cart."""
    cart = cart_store.get_cart(SESSION_ID)
    if product_id in cart.items:
        if cart.items[product_id].quantity > 1:
            cart.items[product_id].quantity -= 1
        else:
            del cart.items[product_id]
    return render_cart(cart)


@rt("/cart/remove/{product_id}")
def remove_from_cart_route(product_id: str):
    """Remove a product completely from cart."""
    cart = cart_store.get_cart(SESSION_ID)
    if product_id in cart.items:
        del cart.items[product_id]
    return render_cart(cart)


@rt("/shop")
def shop(question: str = ""):
    """
    State-aware e-commerce assistant route.

    - GET: render the page with current cart and chat transcript.
    - POST: run the cart_agent with the user's question, update cart via tools,
      append to chat transcript, show updated UI.
    """
    global agent_message_history, chat_transcript

    # Initial GET: no question, just render the page.
    if not question.strip():
        return shop_page()

    deps = CartDeps(cart_store=cart_store, session_id=SESSION_ID)

    try:
        # Run the agent, passing previous message history for multi-turn conversation.
        result = cart_agent.run_sync(
            question,
            deps=deps,
            message_history=agent_message_history,
        )

        # Update message history so the agent has context next time.
        agent_message_history = result.all_messages()

        # Append to UI transcript (simple role/content view).
        chat_transcript.append({"role": "user", "content": question})
        chat_transcript.append({"role": "assistant", "content": result.output})

        usage_obj = result.usage()
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
        }

        return shop_page(usage=usage)

    except Exception as exc:  # demo: surface any error on page
        return shop_page(
            error=f"Something went wrong while running the cart agent: {exc}"
        )


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting FastHTML server...")
    print("Visit: http://localhost:5001/shop")
    serve()
