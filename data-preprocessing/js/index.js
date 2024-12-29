const fs = require("node:fs/promises");
const path = require("node:path");
const { mean } = require("mathjs");
// import { plot } from "nodeplotlib";

// plot(data);

(async () => {
  const file = await fs.readFile(path.resolve(__dirname, "Data.csv"), "utf-8");

  const lines = file.split("\n");
  const headers = lines[0].split(",");
  const dataset = lines.slice(1).map((line) => line.split(","));

  console.log("dataset", dataset);

  let { arrayX, arrayY, uniqueCountries } = dataset.reduce(
    (acc, line) => {
      let [country, age, salary, res] = line;
      age = parseFloat(age);
      salary = parseFloat(salary);
      res = res.toString().trim();
      acc.arrayX.push([country, age, salary]);
      acc.arrayY.push(res === "Yes" ? 1 : 0);
      if (!acc.uniqueCountries.includes(country)) {
        acc.uniqueCountries.push(country);
      }

      return acc;
    },
    { uniqueCountries: [], arrayX: [], arrayY: [] }
  );

  const meanAge = mean(arrayX.map((x) => x[1]).filter((x) => x));
  const meanSalary = mean(arrayX.map((x) => x[2]).filter((x) => x));

  arrayX = arrayX.map((x) => {
    const [country, f1, f2] = x;
    return [country, f1 || meanAge, f2 || meanSalary];
  });

  arrayX = arrayX.map((line) => {
    const [country, ...rest] = line;
    return [...uniqueCountries.map((c) => (c === country ? 1 : 0)), ...rest];
  });

  // convert

  console.log("arrayX", arrayX);
  console.log("arrayY", arrayY);
})();
