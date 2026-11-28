-- Filtro de pandoc: convierte cada ecuación LaTeX en una imagen generada por pdflatex (renderizado LaTeX real).
-- Las imágenes se guardan en ../ecuaciones/ y conservan el tamaño natural de la composición.
local contador = 0
local dir = "../ecuaciones/"

local function medida(pdf)
  local f = io.popen("pdfinfo " .. pdf .. " 2>/dev/null | grep 'Page size'")
  local linea = f:read("*a") or ""
  f:close()
  local w, h = linea:match("([%d%.]+) x ([%d%.]+) pts")
  return tonumber(w), tonumber(h)
end

function Math(el)
  contador = contador + 1
  local nombre = string.format("eq_%03d", contador)
  local tex = el.text:gsub("\\label%b{}", "")
  local es_display = el.mathtype == "DisplayMath"
  if es_display and (tex:find("&") or tex:find("\\\\")) and not tex:find("\\begin{") then
    tex = "\\begin{aligned}" .. tex .. "\\end{aligned}"
  end
  local cuerpo
  if es_display then
    cuerpo = "$\\displaystyle " .. tex .. "$"
  else
    cuerpo = "$" .. tex .. "$"
  end
  local doc = "\\documentclass[11pt,border=1pt]{standalone}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n" .. cuerpo .. "\n\\end{document}\n"
  local tmp = dir .. nombre
  local fh = io.open(tmp .. ".tex", "w"); fh:write(doc); fh:close()
  os.execute("cd " .. dir .. " && pdflatex -interaction=nonstopmode " .. nombre .. ".tex > /dev/null 2>&1; pdftoppm -png -r 300 -singlefile " .. nombre .. ".pdf " .. nombre .. " 2>/dev/null")
  local w, h = medida(tmp .. ".pdf")
  local ancho = (w or 40) / 72.0
  local alto = (h or 14) / 72.0
  local img = pandoc.Image({}, tmp .. ".png", "", pandoc.Attr("", {}, {width = string.format("%.4fin", ancho), height = string.format("%.4fin", alto)}))
  return img
end
