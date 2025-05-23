const chatInput = document.getElementById("chat-input");
const chatForm = document.getElementById("chat-form");
const chatbotFigure = document.querySelector(".mobile");

let apiURL = "";

function toggleChatBot() {
  chatbotFigure.classList.toggle("hidden");
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();

  let userMsg = chatInput.value;
  addMessage(userMsg, "outgoing");

  fetch(`http://127.0.0.1:5000/chatbot_api/${apiURL}`, {
    method: "POST",
    body: JSON.stringify({ message: userMsg }),
    mode: "cors",
    headers: { "Content-Type": "application/json" },
  })
    .then((r) => r.json())
    .then((r) => {
      if (r.url) {
        apiURL = r.url;
      } else {
        apiURL = "";
      }
      // Pass the entire response to addMessage to handle media and links
      addMessage(r.response, "incoming", r.media, r.link);

      if (r.data) {
        addPDFBtn(r.data);
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      addMessage("Error occurred while fetching response.", "incoming");
    });
});

function addMessage(message, msgtype, media = [], link = "") {
  const chatMessage = document.createElement("div");
  chatMessage.classList.add("chat-message");
  chatMessage.classList.add(`${msgtype}-message`);

  // Handle text message
  const textDiv = document.createElement("div");
  const linkRegex = /<a\s+href=[\'"]?([^"\'>]+)[\'"]?>/i;

  if (linkRegex.test(message)) {
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = message;
    const linkElement = tempDiv.querySelector("a");

    if (linkElement) {
      textDiv.appendChild(linkElement);
      linkElement.addEventListener("click", (event) => {
        event.preventDefault();
        window.open(linkElement.href, "_blank");
      });
    } else {
      textDiv.innerHTML = message;
    }
  } else {
    textDiv.innerHTML = message;
  }
  chatMessage.appendChild(textDiv);

  // Handle media (images and PDFs)
  if (media && media.length > 0) {
    media.forEach((item) => {
      if (item.type === "image") {
        const img = document.createElement("img");
        img.src = `/${item.path}`;
        img.alt = item.alt;
        img.style.maxWidth = "100%";
        img.style.height = "auto";
        img.style.marginTop = "10px";
        chatMessage.appendChild(img);
      } else if (item.type === "pdf") {
        const pdfLink = document.createElement("div");
        pdfLink.classList.add("file-message");
        pdfLink.innerHTML = item.name;
        pdfLink.style.marginTop = "10px";
        pdfLink.style.cursor = "pointer";
        pdfLink.style.color = "blue";
        pdfLink.style.textDecoration = "underline";
        pdfLink.onclick = () => {
          window.open(`/${item.path}`, "_blank");
        };
        chatMessage.appendChild(pdfLink);
      }
    });
  }

  // Handle link field if present
  if (link) {
    const linkDiv = document.createElement("div");
    linkDiv.innerHTML = link;
    const linkElement = linkDiv.querySelector("a");
    if (linkElement) {
      linkElement.addEventListener("click", (event) => {
        event.preventDefault();
        window.open(linkElement.href, "_blank");
      });
    }
    chatMessage.appendChild(linkDiv);
  }

  document.querySelector(".chat-messages").appendChild(chatMessage);
  document.querySelector(".chat-messages").scrollTop +=
    chatMessage.getBoundingClientRect().y + 20;
  chatInput.value = "";
}

function addPDFBtn(data) {
  const chatMessage = document.createElement("div");
  chatMessage.classList.add("chat-message");
  chatMessage.classList.add("incoming-message");
  chatMessage.classList.add("file-message");
  chatMessage.innerHTML = data.filename;
  chatMessage.style.cursor = "pointer";
  chatMessage.style.color = "blue";
  chatMessage.style.textDecoration = "underline";
  chatMessage.onclick = (e) => {
    window.open(data.link, "_blank");
  };
  document.querySelector(".chat-messages").appendChild(chatMessage);
  document.querySelector(".chat-messages").scrollTop +=
    chatMessage.getBoundingClientRect().y + 10;
  chatInput.value = "";
}
