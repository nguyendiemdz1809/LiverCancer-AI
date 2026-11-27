import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";

import { getHealth } from "./api";

import Home from "./pages/Home";
import Analyze from "./pages/Analyze";
import Models from "./pages/Models";
import Research from "./pages/Research";
import About from "./pages/About";
import FollowUp from "./pages/FollowUp";

const ICON: Record<string, string> = {
  home: "M3 11l9-7 9 7M5 10v10h5v-6h4v6h5V10",
  scan: "M4 5h16v14H4zM4 15l4-4 4 4 3-3 5 5M15 9h.01",
  model: "M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zM12 12l8-4.5M12 12v9M12 12L4 7.5",
  chart: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  info: "M12 8h.01M11 12h1v5h1M12 3a9 9 0 100 18 9 9 0 000-18z",
  folder: "M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7zM8 13h8M12 10v6",
};
const LINKS = [
  { to: "/", label: "Trang chủ", end: true, icon: "home" },
  { to: "/phan-tich", label: "Phân tích ảnh", icon: "scan" },
  { to: "/theo-doi", label: "Hồ sơ theo dõi", icon: "folder" },
  { to: "/mo-hinh", label: "Thông tin mô hình", icon: "model" },
  { to: "/ket-qua", label: "Kết quả nghiên cứu", icon: "chart" },
  { to: "/gioi-thieu", label: "Giới thiệu đề tài", icon: "info" },
];

const TITLES: Record<string, string> = {
  "/": "LiverCancer AI", "/phan-tich": "Phân tích ảnh · LiverCancer AI", "/theo-doi": "Hồ sơ theo dõi · LiverCancer AI", "/mo-hinh": "Thông tin mô hình · LiverCancer AI",
  "/ket-qua": "Kết quả nghiên cứu · LiverCancer AI",
  "/gioi-thieu": "Giới thiệu đề tài · LiverCancer AI",
};

function NavIcon({ name }: { name: string }) {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICON[name]} />
    </svg>
  );
}

// Live state of the AI server. It runs on the research team's computer (not 24/7), so every page says whether
// image analysis is available right now.
function AiStatus() {
  const [st, setSt] = useState<"checking" | "ready" | "loading" | "off" | "error">("checking");
  const [device, setDevice] = useState("");
  useEffect(() => {
    let alive = true, timer: number | undefined;
    const poll = async () => {
      let next = 30000;
      try {
        const h = await getHealth();
        const v = Object.values(h.models);
        const s2 = v.some((x) => x === "loading") ? "loading" : v.every((x) => x === "error") ? "error" : "ready";
        if (!alive) return;
        setSt(s2); setDevice(h.device);
        if (s2 === "loading") next = 4000;
      } catch {
        if (alive) setSt("off");
      }
      if (alive) timer = window.setTimeout(poll, next);
    };
    poll();
    return () => { alive = false; window.clearTimeout(timer); };
  }, []);
  const short = {
    checking: "Máy chủ AI: Đang kiểm tra…",
    ready: `Máy chủ AI: Đang hoạt động${device ? ` (${device})` : ""}`,
    loading: "Máy chủ AI: Đang khởi động",
    off: "Máy chủ AI: Đang tắt",
    error: "Máy chủ AI: Lỗi tải mô hình",
  }[st];
  const long = {
    checking: "Đang kiểm tra máy chủ AI.",
    ready: "Máy chủ AI đang hoạt động, có thể phân tích ảnh.",
    loading: "Máy chủ AI đang khởi động mô hình, vui lòng chờ.",
    off: "Máy chủ AI đang tắt. Phân tích ảnh chỉ dùng được khi máy tính chạy AI của nhóm nghiên cứu được bật; các trang thông tin vẫn xem được.",
    error: "Máy chủ AI lỗi khi tải mô hình, tạm thời không phân tích được ảnh.",
  }[st];
  return (
    <div className={`ai-status ${st}`} role="status" aria-live="polite" title={long}>
      <span className="dot" aria-hidden="true" /><span className="ai-text">{short}</span><span className="sr-only">. {long}</span>
    </div>
  );
}

function Header() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo(0, 0); document.title = TITLES[loc.pathname] || "LiverCancer AI"; }, [loc.pathname]);
  return (
    <header className="site-header">
      <div className="container wide nav">
        <NavLink to="/" className="brand" aria-label="LiverCancer AI – Trang chủ">
          <img src="/favicon.svg" alt="" />
          <span className="brand-text"><span className="brand-name">LiverCancer <b>AI</b></span><span className="brand-sub">Hỗ trợ phát hiện và phân loại ung thư gan</span></span>
        </NavLink>
        <nav id="nav-links" className={`nav-links${open ? " open" : ""}`} aria-label="Điều hướng chính">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => (isActive ? "active" : "")}><NavIcon name={l.icon} />{l.label}</NavLink>
          ))}
        </nav>
        <AiStatus />
        <button className="menu-btn" aria-expanded={open} aria-controls="nav-links" onClick={() => setOpen(!open)}>
          {open ? "Đóng" : "Menu"}
        </button>
      </div>
    </header>
  );
}

// The AI server address is not a user setting: /runtime-config.json on Cloudflare Pages (written by MO_WEBSITE_ONLINE),
// VITE_API_BASE at build time, else the same address as the site (see docs/DEPLOY_CLOUDFLARE.md).
function Footer() {
  return (
    <footer className="site-footer">
      <div className="container wide">
        <div>
          <b>LiverCancer AI</b> · Đồ án tốt nghiệp: ứng dụng học sâu trong phát hiện và phân loại ung thư gan từ ảnh CT/MRI.
          <br />Hệ thống nghiên cứu, <b>không thay thế chẩn đoán của bác sĩ</b>. Ảnh tải lên không được lưu trữ.
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <>
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/phan-tich" element={<Analyze />} />
          <Route path="/theo-doi" element={<FollowUp />} />
          <Route path="/mo-hinh" element={<Models />} />
          <Route path="/ket-qua" element={<Research />} />
          <Route path="/gioi-thieu" element={<About />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}
