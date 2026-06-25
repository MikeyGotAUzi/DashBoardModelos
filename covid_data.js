/* =====================================================
   BASE DE DATOS CLINICA SIMULADA — COVID-19
   300 registros de pacientes
   Columnas:
     edad, genero, dx, intervenciones, hospitalizado,
     urgencias, gravedad,
     tubo (0/1), estancia (dias), mortalidad (0/1)
   ===================================================== */

function generarDatosCovid() {
  const data = [];
  const seed = (n) => {
    // generador pseudo-aleatorio determinista
    let x = Math.sin(n + 1) * 10000;
    return x - Math.floor(x);
  };

  for (let i = 0; i < 300; i++) {
    const r  = (offset) => seed(i * 17 + offset);

    const edad           = Math.floor(r(1) * 90) + 5;          // 5-94
    const genero         = r(2) > 0.5 ? 1 : 0;                 // 1=M, 0=F
    const dx             = Math.floor(r(3) * 3);                // 0,1,2
    const intervenciones = Math.floor(r(4) * 8);                // 0-7
    const hospitalizado  = r(5) > 0.3 ? 1 : 0;
    const urgencias      = r(6) > 0.5 ? 1 : 0;
    const gravedad       = Math.floor(r(7) * 5) + 1;            // 1-5

    // reglas clinicas para las etiquetas
    const riesgoBase = (edad / 100) * 0.4
      + (gravedad / 5) * 0.35
      + (intervenciones / 8) * 0.15
      + (urgencias * 0.1);

    const tubo       = riesgoBase + r(8) * 0.15 > 0.45 ? 1 : 0;
    const estancia   = Math.max(1, Math.round(
      2 + gravedad * 2.5 + intervenciones * 0.8 + (edad > 60 ? 3 : 0) + r(9) * 3
    ));
    const mortalidad = riesgoBase + r(10) * 0.1 > 0.6 ? 1 : 0;

    data.push([
      edad, genero, dx, intervenciones,
      hospitalizado, urgencias, gravedad,
      tubo, estancia, mortalidad
    ]);
  }
  return data;
}

const COVID_DATA = generarDatosCovid();
